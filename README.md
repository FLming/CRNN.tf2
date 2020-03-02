# Convolutional Recurrent Neural Network for End-to-End Text Recognize - TensorFlow 2

This is a re-implementation of [CRNN](http://arxiv.org/abs/1507.05717) network, build by tensorflow 2. This repository allows you to understand how to build an end-to-end text recognition network in a simple way. By the way, Here is official [repo](https://github.com/bgshih/crnn).

**I am building a [EAST network for scene text detection by tensorflow 2](https://github.com/FLming/EAST.tf2), if you are interested, welcome to build together**

## Abstract

### Requirements

```
tensorflow >= 2.0.0
```

### Features

- Easy to understand
- Easy to change the backbone
- Easy to use other components of TensorFlow, such as serving
- [x] Tensorflow serving
- [ ] Tensorflow lite
- [ ] Distributed training

This repo aims to build a simple, efficient, end-to-end text recognize network by using the various components of tensorflow 2.

## Data prepare

In order to train this network, you should prepare a lookup table, images and labels. Example is copy from [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/).

### [Lookup table](./example/table.txt)

The file contains all characters and blank label (in the last or any place both ok, but I find tf decoders can't change it, so set it to last). Currently, you can write any word for blank.

### Image data

The inputs of image height must equals 32 because of CNN construction. If you want to change it, you have to change the construction of CNN.

![Paintbrushes](example/images/1_Paintbrushes_55044.jpg)
![Reimbursing](example/images/2_Reimbursing_64165.jpg)
![Creationisms](example/images/3_Creationisms_17934.jpg)

### [Lable data](./example/annotation.txt)

The format of annotation file can like:
```
[Relative position of the image file] [lable]
```
or MJSynth format or any format you want. see read_imagepaths_and_labels function in dataset.py file and add new parse for you data.

### Nets

Network structure can be viewed at doc folder.


## Train

```bash
python train.py -ta /PATH/TO/TXT -va /PATH/TO/TXT -tf the name of parse funcs -vf the name of parse funcs -t /PATH/TO/TABLE ...
```
Example:
```
python train.py -ta /data/mnt/ramdisk/max/90kDICT32px/annotation_train.txt -va /data/mnt/ramdisk/max/90kDICT32px/annotation_val.txt -tf mjsynth mjsynth -va /data/mnt/ramdisk/max/90kDICT32px/annotation_test.txt -vf mjsynth -t example/table.txt -e 30 --max_to_keep 25
```
The name of parse funcs I provide is mjsynth, icdar2013 and example.

For other parameters please check the `train.py -h`

The training process can be viewed using tensorboard

```bash
tensorboard --logdir=logs/
```

![tensorboard](doc/tensorboard.png)

## Eval

```bash
python eval.py -a /PATH/TO/TXTs -f the name of parse funcs -t /PATH/TO/TABLE -c /PATH/TO/CHECKPOINT
```

For other parameters please check the `eval.py -h`

## Converte

Before you depoly or demo, you should pick up a good weight, and use converter to make a h5 file or SavedModel
```
python converter.py -t /PATH/TO/TABLE -c /PATH/TO/CHECKPOINT -f tf/h5 -o /PATH/TO/OUTPUT 
```

## Demo inference

```bash
python demo.py -i /PATH/TO/images/ -t /PATH/TO/TABLE --model /PATH/TO/MODEL
```

then, You will see output:
```
Path: 1_Paintbrushes_55044.jpg
        Greedy: Paintbrushes
        Beam search: Paintbrushes
Path: 2_Reimbursing_64165.jpg
        Greedy: Reimbursing
        Beam search: Reimbursing
Path: 3_Creationisms_17934.jpg
        Greedy: Creationisms
        Beam search: Creationisms
```

## Tensorflow serving

Please refer to the official website for [installation](https://www.tensorflow.org/tfx/serving/setup).

Just run tensorflow serving by 
```bash
tensorflow_model_server --rest_api_port=8501 --model_name=CRNN --model_base_path="/path/to/SavedModel/"
```