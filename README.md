# Convolutional Recurrent Neural Network for End-to-End Text Recognize - TensorFlow 2

This is a re-implementation of [CRNN](http://arxiv.org/abs/1507.05717) network, build by tensorflow 2. This repository may help you to understand how to build an end-to-end text recognition network in a simple way. By the way, Here is official [repo](https://github.com/bgshih/crnn).

## Abstract

### Requirements

```
tensorflow >=2.2, if you use Tensorflow 2.0, 2.1, you can check the custom_training_loop branch.
```

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

### [Label data](./example/annotation.txt)

The format of annotation file is:

1. example format: `XX.jpg label`
2. mjsynth format: `XX_label_XX.jpg XX`
3. icdar2013 format: `XX.png, "label"`

or any format you want. see read_imagepaths_and_labels function in dataset.py file and add new parse for you data.

### Nets

Network structure can be viewed at doc folder.


## Train

```bash
python train.py -ta /PATH/TO/TXT -va /PATH/TO/TXT -tf the name of parse funcs -vf the name of parse funcs -t /PATH/TO/TABLE ...
```
Example:
```
python train.py -ta /data/mnt/ramdisk/max/90kDICT32px/annotation_train.txt /data/mnt/ramdisk/max/90kDICT32px/annotation_val.txt -tf mjsynth mjsynth -va /data/mnt/ramdisk/max/90kDICT32px/annotation_test.txt -vf mjsynth -t example/table.txt -e 30
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
python eval.py -a /PATH/TO/TXTs -f the name of parse funcs -t /PATH/TO/TABLE -m /PATH/TO/MODEL
```

For other parameters please check the `eval.py -h`

## Demo inference

Here I provide a model that trained on Mjsytch, this model can only predict 0-9, a-z and A-Z.

- [baidu](https://pan.baidu.com/s/1j49KO0AJpVWQ94Yps-yYNw), code is `hhvm`
- [google](https://drive.google.com/open?id=1qUoH3U86YwmsbRCt7vw8WwpZMXKRwDSp)

```bash
python demo.py -i example/images/ -t example/table.txt -m model/
```

then, You will see output like this:
```
Path: example/images/1_Paintbrushes_55044.jpg, greedy: Paintbrushes, beam search: Paintbrushes
Path: example/images/2_Reimbursing_64165.jpg, greedy: Reimbursing, beam search: Reimbursing
Path: example/images/3_Creationisms_17934.jpg, greedy: Creationisms, beam search: Creationisms
```

## Converte

Before you depoly, you should pick up a good weight, and converte model to SavedModel format in order to use other tensorflow components(etc. `Tensorflow serving`、`Tensorflow.js`...)
```
python converter.py --model /PATH/TO/MODEL -o /PATH/TO/OUTPUT
```