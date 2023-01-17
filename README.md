# efficientdet_automl_IPU
IPU porting efficientdet-d2

## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
``` 
python3 -m venv <venv name>   
source <venv path>/bin/activate
```
2. Navigate to the Poplar SDK root directory

3. Install the Tensorflow2 and IPU Tensorflow add-ons wheels:

```
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Navigate to this example's root directory

5. Install the Python requirements with:

` pip3 install -r requirements.txt `

## dataset & checkpoints
Download data and checkpoints.

```
# Download and convert pascal data.
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar xf VOCtrainval_11-May-2012.tar

!PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
    --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal

# Download backbone checkopints.
!wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2.tar.gz
!tar xf efficientdet-d2.tar.gz
```

## Run
Executes without specifying any additional parameters
```
python ipu_train.py
```