# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The main training script."""
import os
import platform
from absl import app
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod

import re
import dataloader
import hparams_config
import utils
from tf2 import train_lib
from tf2 import util_keras
from tf2 import efficientdet_keras
from tensorflow.python import ipu
import ipu_train_lib
from ipu_automl_io import (
    postprocess_predictions,
    preprocess_normalize_image,
    visualise_detections,
)
from ipu_utils.dataset import (
    get_dataset,
    input_tensor_shape,
)
from ipu_utils import (
    create_app_json,
    preload_fp32_weights,
    load_weights_into_model,
    set_or_add_env,
)

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import math


# Configure the IPU device.
ipu_config = ipu.config.IPUConfig()

ipu_config.convolutions.poplar_options['partialsType'] = "half"
ipu_config.matmuls.poplar_options['partialsType'] = "half"

ipu_config.auto_select_ipus = 4
ipu_config.device_connection.enable_remote_buffers = True
ipu_config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
ipu_config.configure_ipu_system()

outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(buffer_depth=3)

model_name = "efficientdet-d2"
model_dir = "./efficientdet-d2-finetune"
hparams = "./voc_config.yaml"
train_file_pattern = "./tfrecord/pascal*.tfrecord"
val_file_pattern = "./tfrecord/pascal*.tfrecord"
val_json_file = "./tfrecord/*.json"
num_examples_per_epoch = 5717
num_epochs = 1
steps_per_execution = 10
batch_size = 1
eval_samples = 1
max_instances_per_image = 100
num_iterations = 20

micro_batch_size = 1
io_precision = tf.float16

class CosineLRSchedule(LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate: float,
                 weight_updates_to_total_decay: int):

        super(CosineLRSchedule, self).__init__()
        self._weight_updates_to_total_decay = float(weight_updates_to_total_decay)
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step = tf.minimum(step, self._weight_updates_to_total_decay)
        lr = self.initial_learning_rate * 0.5 * (1 + tf.cos((step * math.pi) / self._weight_updates_to_total_decay))
        return lr

def init_experimental(config):
  """Serialize train config to model directory."""
  tf.io.gfile.makedirs(config.model_dir)
  config_file = os.path.join(config.model_dir, 'config.yaml')
  if not tf.io.gfile.exists(config_file):
    tf.io.gfile.GFile(config_file, 'w').write(str(config))

def _detection_loss(cls_outputs, box_outputs, labels, loss_vals, in_model,
                                positives_momentum, min_level, max_level, num_classes,
                                data_format, alpha, gamma, label_smoothing,
                                delta, box_loss_weight, iou_loss_type, num_scales,
                                aspect_ratios, anchor_scale, image_size, iou_loss_weight):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundtruth targets.
      loss_vals: A dict of loss values.

    Returns:
      total_loss: an integer tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integer tensor representing total class loss.
      box_loss: an integer tensor representing total box regression loss.
      box_iou_loss: an integer tensor representing total box iou loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    dtype = cls_outputs[0].dtype
    num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
    positives_momentum = positives_momentum or 0
    if positives_momentum > 0:
      # normalize the num_positive_examples for training stability.
      moving_normalizer_var = tf.Variable(
          0.0,
          name='moving_normalizer',
          dtype=dtype,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)
      num_positives_sum = tf.keras.backend.moving_average_update(
          moving_normalizer_var,
          num_positives_sum,
          momentum=positives_momentum)
    elif positives_momentum < 0:
      num_positives_sum = utils.cross_replica_mean(num_positives_sum)
    num_positives_sum = tf.cast(num_positives_sum, dtype)
    levels = range(len(cls_outputs))
    cls_losses = []
    box_losses = []
    for level in levels:
      # Onehot encoding for classification labels.
      cls_targets_at_level = tf.one_hot(
          labels['cls_targets_%d' % (level + min_level)],
          num_classes,
          dtype=dtype)

      if data_format == 'channels_first':
        bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, -1, width, height])
      else:
        bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, width, height, -1])

      # class_loss_layer = in_model.loss.get(FocalLoss.__name__, None)
      class_loss_layer = ipu_train_lib.FocalLoss(alpha, gamma, label_smoothing, reduction=tf.keras.losses.Reduction.NONE)
      if class_loss_layer:
        cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                    cls_outputs[level])
        if data_format == 'channels_first':
          cls_loss = tf.reshape(
              cls_loss, [bs, -1, width, height, num_classes])
        else:
          cls_loss = tf.reshape(
              cls_loss, [bs, width, height, -1, num_classes])
        cls_loss *= tf.cast(
            tf.expand_dims(
                tf.not_equal(
                    labels['cls_targets_%d' % (level + min_level)],
                    -2), -1), dtype)
        cls_loss_sum = tf.reduce_sum(cls_loss)
        cls_losses.append(tf.cast(cls_loss_sum, dtype))

      box_loss_class = ipu_train_lib.BoxLoss(delta, reduction=tf.keras.losses.Reduction.NONE)
      if box_loss_weight and box_loss_class:
        box_targets_at_level = (
            labels['box_targets_%d' % (level + min_level)])
        box_loss_layer = box_loss_class
        box_losses.append(
            box_loss_layer([num_positives_sum, box_targets_at_level],
                           box_outputs[level]))

    if iou_loss_type:
      box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs],
                              axis=0)
      box_targets = tf.concat([
          tf.reshape(labels['box_targets_%d' %
                            (level + min_level)], [-1, 4])
          for level in levels
      ],
                              axis=0)
      # box_iou_loss_layer = self.loss[BoxIouLoss.__name__]
      box_iou_loss_layer = ipu_train_lib.BoxIouLoss(iou_loss_type, min_level, max_level, 
                        num_scales, aspect_ratios, anchor_scale, image_size)
      box_iou_loss = box_iou_loss_layer([num_positives_sum, box_targets],
                                        box_outputs)
      loss_vals['box_iou_loss'] = box_iou_loss
    else:
      box_iou_loss = 0

    cls_loss = tf.add_n(cls_losses) if cls_losses else 0
    box_loss = tf.add_n(box_losses) if box_losses else 0
    total_loss = (
        cls_loss + box_loss_weight * box_loss +
        iou_loss_weight * box_iou_loss)
    loss_vals['det_loss'] = total_loss
    loss_vals['cls_loss'] = cls_loss
    loss_vals['box_loss'] = box_loss

    return total_loss

def _reg_l2_loss(var_freeze_expr, in_model, weight_decay, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n([
      tf.nn.l2_loss(v) for v in _freeze_vars(var_freeze_expr, in_model) if var_match.match(v.name)
  ])

def _freeze_vars(var_freeze_expr, in_model):
  if var_freeze_expr:
    return [
        v for v in in_model.trainable_variables
        if not re.match(var_freeze_expr, v.name)
    ]
  return in_model.trainable_variables

#
# A custom training loop
#
@tf.function(experimental_compile=True)
# def train_step(config, in_model, data, optimizer):
def train_step(in_model, data, optimizer, steps_per_execution, weight_decay, clip_gradients_norm, var_freeze_expr,
                                positives_momentum, min_level, max_level, num_classes,
                                data_format, alpha, gamma, label_smoothing,
                                delta, box_loss_weight, iou_loss_type, num_scales,
                                aspect_ratios, anchor_scale, image_size, iou_loss_weight):
    """Train step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """

    for _ in tf.range(steps_per_execution) :
      images, labels = next(data)
          
      with tf.GradientTape() as tape:
        cls_outputs, box_outputs = util_keras.fp16_to_fp32_nested(in_model(images, training=True))
        loss_dtype = cls_outputs[0].dtype

        labels = util_keras.fp16_to_fp32_nested(labels)

        total_loss = 0
        loss_vals = {}
        det_loss = _detection_loss(cls_outputs, box_outputs, labels, loss_vals, in_model,
                                positives_momentum, min_level, max_level, num_classes,
                                data_format, alpha, gamma, label_smoothing,
                                delta, box_loss_weight, iou_loss_type, num_scales,
                                aspect_ratios, anchor_scale, image_size, iou_loss_weight)
        total_loss += det_loss

        reg_l2_loss = _reg_l2_loss(var_freeze_expr, in_model, weight_decay)
        loss_vals['reg_l2_loss'] = reg_l2_loss
        total_loss += tf.cast(reg_l2_loss, loss_dtype)

        # scaled_loss = total_loss

        grads = tape.gradient(total_loss, in_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, in_model.trainable_variables))

      # loss_vals['loss'] = total_loss
      # print(f'total_loss : {total_loss}')
      # loss_vals['learning_rate'] = optimizer.learning_rate(optimizer.iterations)
      # trainable_vars = _freeze_vars(var_freeze_expr, in_model)
      # scaled_gradients = tape.gradient(total_loss, trainable_vars)

      # gradients = scaled_gradients
      # if clip_gradients_norm > 0:
      #   clip_norm = abs(clip_gradients_norm)
      #   gradients = [
      #       tf.clip_by_norm(g, clip_norm) if g is not None else None
      #       for g in gradients
      #   ]
      #   gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
      #   loss_vals['gradient_norm'] = tf.linalg.global_norm(gradients)
      # optimizer.apply_gradients(zip(scaled_gradients, in_model.trainable_variables))
      # # return loss_vals
      # print(f'loss_vals : {loss_vals}')
      outfeed_queue.enqueue(total_loss)


def main():
  # Parse and override hparams
  config = hparams_config.get_detection_config(model_name)
  config.override(hparams)
  config.num_epochs = num_epochs
  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  steps_per_epoch = num_examples_per_epoch // batch_size
  params = dict(
      profile=False,
      model_name=model_name,
      steps_per_execution=steps_per_execution,
      model_dir=model_dir,
      steps_per_epoch=steps_per_epoch,
      batch_size=batch_size,
      tf_random_seed=None,
      debug=False,
      val_json_file=val_json_file,
      eval_samples=eval_samples)
  config.override(params, True)

  def model_fn(in_shape, training=True):
    inputs = layers.Input(
        in_shape[1:], batch_size=micro_batch_size, dtype=io_precision)
    cast_input = preprocess_normalize_image(
        inputs, tf.float16)

    detnet = efficientdet_keras.EfficientDetNet(config=config)
    print(f'detnet.layers : {detnet.layers}')
    outputs = detnet(cast_input, training=training)

    return inputs, outputs

  def get_dataset(is_training, config):
    file_pattern = (
        train_file_pattern
        if is_training else val_file_pattern)
    if not file_pattern:
      raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=False,
        max_instances_per_image=max_instances_per_image,
        debug=config.debug)(
            config.as_dict())

  in_shape = (micro_batch_size, ) + \
    input_tensor_shape(config.image_size)
  #
  # Execute the graph
  #
  ipu_strategy = ipu.ipu_strategy.IPUStrategy()
  #stage = [1, 3, 4] #, 5]
  stage = [1,2,4, -1]
  
  with ipu_strategy.scope():
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build((None, *config.image_size, 3))

    # Get a blank set of pipeline stage assignments.
    # optimizer=train_lib.get_optimizer(config.as_dict())
    learning_rate = CosineLRSchedule(config.learning_rate, config.weight_decay)
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=config.momentum)

    if tf.train.latest_checkpoint(model_dir):
      ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
      util_keras.restore_ckpt(
          model,
          ckpt_path,
          config.moving_average_decay)

    init_experimental(config)
    assignments = model.get_pipeline_stage_assignment()

    assignment_flag = 0
    for i, assignment in enumerate(assignments):
      if stage[assignment_flag] == i :
        assignment_flag +=1 
        assignment.pipeline_stage = assignment_flag
      else :
        assignment.pipeline_stage = assignment_flag


    # Apply the modified assignments back to the model.
    model.set_pipeline_stage_assignment(assignments)
    model.set_pipelining_options(gradient_accumulation_steps_per_replica=2)
    model.print_pipeline_stage_assignment_summary()
    model.summary()
    
    train_iterator = iter(get_dataset(True, config))
    for begin_step in range(0, num_iterations, config.steps_per_execution):
        ipu_strategy.run(train_step, args=[model, train_iterator, optimizer, 
                                                    config.steps_per_execution, config.weight_decay, config.clip_gradients_norm, config.var_freeze_expr,
                                                    config.positives_momentum, config.min_level, config.max_level, config.num_classes,
                                                    config.data_format, config.alpha, config.gamma, config.label_smoothing,
                                                    config.delta, config.box_loss_weight, config.iou_loss_type, config.num_scales,
                                                    config.aspect_ratios, config.anchor_scale, config.image_size, config.iou_loss_weight])
        print(f'outfeed_queue : {outfeed_queue}')
        mean_loss = sum(outfeed_queue) / config.steps_per_execution
        print(f'mean_loss : {mean_loss}')


if __name__ == '__main__':
    main()
