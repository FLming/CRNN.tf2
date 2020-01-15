# Convolutional Recurrent Neural Network for End-to-End Text Recognize - TensorFlow 2

This repo is a implement of [CRNN](http://arxiv.org/abs/1507.05717), [authors repo](https://github.com/bgshih/crnn).
And this net have VGG and ResNet backbone([reference](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)).

## Abstract

### Requirements

```
python >= 3.6
tensorflow >= 2.0.0
```

This repo aims to build a efficient, complete end-to-end text recognize network only by using the various components of tensorflow 2.

## Data prepare

In order to train this network, you should prepare a lookup table, images and labels. Example data both on example folder(copy from [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)).

### [Lookup table](./example/table.txt)

See example. The file contains all characters and blank label(in the last or any place both ok, but I find tf decoders set it to last and can't change it.) Currently, you can write any word for blank.

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
or MJSynth format or any format you want. see read_imagepaths_and_labels function.

### Nets

Add backbone([VGG](doc/VGG_CRNN.png) or [ResNet](doc/ResNet_CRNN.png)) arg to train can change the backbone of the net.
Network structure can be viewed at doc folder.


## Train

```bash
python train.py -ta /PATH/TO/TXT -va /PATH/TO/TXT -t /PATH/TO/TABLE
```

For other parameters please check the `train.py -h`

The training process can be viewed using tensorboard

```bash
tensorboard --logdir=tensorboard/
```

![tensorboard](doc/tensorboard.png)

## Eval

```bash
python eval.py -a /PATH/TO/TXT -t /PATH/TO/TABLE --checkpoint /PATH/TO/CHECKPOINT
```

For other parameters please check the `eval.py -h`

## Demo inference

```bash
python demo.py -i example/images/ -t example/table.txt --model /PATH/TO/MODEL
```

then, You will see output:
```
*************** Greedy ***************
Path: 1_Paintbrushes_55044.jpg, prediction: Paintbrushes
Path: 2_Reimbursing_64165.jpg, prediction: Reimbursing
Path: 3_Creationisms_17934.jpg, prediction: Creationisms
*************** Beam search ***************
Path: 1_Paintbrushes_55044.jpg, prediction: Paintbrushes
Path: 2_Reimbursing_64165.jpg, prediction: Reimbursing
Path: 3_Creationisms_17934.jpg, prediction: Creationisms
```

## Tensorflow serving

Please refer to the official website for [installation](https://www.tensorflow.org/tfx/serving/setup).

1. First you should pick a good model.
2. convert checkpoint to SavedModel by converter.py
3. Just run tensorflow serving by 
```bash
tensorflow_model_server --rest_api_port=8501 --model_name=CRNN --model_base_path="/path/to/SavedModel/"
```