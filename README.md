# Convolutional Recurrent Neural Network for End-to-End Text Recognize - TensorFlow 2

![TensorFlow version](https://img.shields.io/badge/TensorFlow->=2.2-FF6F00?logo=tensorflow)
![Python version](https://img.shields.io/badge/Python->=3.6-3776AB?logo=python)
[![Paper](https://img.shields.io/badge/paper-arXiv:1507.05717-B3181B?logo=arXiv)](https://arxiv.org/abs/1507.05717)
[![Zhihu](https://img.shields.io/badge/知乎-文本识别网络CRNN—实现简述-blue?logo=zhihu)](https://zhuanlan.zhihu.com/p/122512498)

This is a re-implementation of the CRNN network, build by TensorFlow 2. This repository may help you to understand how to build an End-to-End text recognition network easily. Here is the official [repo](https://github.com/bgshih/crnn) implemented by [bgshih](https://github.com/bgshih).

## Abstract

This repo aims to build a simple, efficient text recognize network by using the various components of TensorFlow 2. The model build by the Keras API, the data pipeline build by `tf.data`, and training with `model.fit`, so we can use most of the functions provided by TensorFlow 2, such as `Tensorboard`, `Distribution strategy`, `TensorFlow Profiler` etc.

## Demo

Here I provide an example model that trained on the Mjsynth dataset, this model can only predict 0-9 and a-z(ignore case).

- [百度, 提取码: g7dj](https://pan.baidu.com/s/1Gx29JwtQ4HX_53gUajHOAg)
- [Google](https://drive.google.com/open?id=1gTJ6Fgo7sfCJdA5ZUBkB76GtcC6Owqly)

```bash
$ python crnn/demo.py -i example/images/ -t exmaple/table.txt -m PATH/TO/MODEL
```

Then, You will see output like this:
```
Path: example/images/1_Paintbrushes_55044.jpg, greedy: paintbrushes, beam search: paintbrushes
Path: example/images/3_Creationisms_17934.jpg, greedy: creationisms, beam search: creationisms
Path: example/images/2_Reimbursing_64165.jpg, greedy: reimbursing, beam search: reimbursing
```

Sometimes the beam search method will be better than the greedy method, but it's costly.

## Train

Before you start training, maybe you should [prepare](#Data-prepare) data first. All predictable characters are defined by the [table.txt](example/table.txt) file. The configuration of the training process is defined by the [yaml](configs/mjsynth.yaml) file.

### Installation

```bash
$ pip install -r requirements.txt
```

This training script uses all GPUs by default, if you want to use a specific GPU, please set the `CUDA_VISIBLE_DEVICES` parameter.

```bash
$ export CUDA_VISIBLE_DEVICES=0
$ python crnn/train.py --config configs/mjsynth.yaml --model_dir PATH/TO/SAVE
```

The training process can visualize in Tensorboard. 

```bash
$ tensorboard --logdir PATH/TO/MODEL_DIR
```

For more instructions, please refer to the [yaml](configs/mjsynth.yaml) file.

## Data prepare

To train this network, you should prepare a lookup table, images and corresponding labels. Example data is copy from [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=introduction) dataset.

### [Lookup table](./example/table.txt)

The file contains all characters and blank labels (in the last or any place both ok, but I find Tensorflow decoders can't change it now, so set it to last). By the way, you can write any word as blank.

### Image data

It's an End-to-End method, so we don't need to indicate the position of the character in the image.

![Paintbrushes](example/images/1_Paintbrushes_55044.jpg)
![Creationisms](example/images/3_Creationisms_17934.jpg)
![Reimbursing](example/images/2_Reimbursing_64165.jpg)

The labels corresponding to these three pictures are `Paintbrushes`, `Creationisms`, `Reimbursing`.

### Annotation file

We should write the image path and its corresponding label to a text file in a certain format such as example data. The data input pipeline will automatically detect the support format. Customization is also very simple, please check out the [dataset factory](crnn/dataset_factory.py).

#### Support format

- [MJSynth](./example/mjsynth_annotation.txt)
- [ICDAR2013/2015](./example/icdar2013_annotation.txt)
- [Simple](./example/simple_annotation.txt) such as [example.jpg label]

## Eval

```bash
$ python crnn/eval.py --config PATH/TO/CONFIG_FILE --model PATH/TO/MODEL
```

## Converte & Ecosystem

There are many components here to help us do other things. For example, deploy by `Tensorflow serving`. Before you deploy, you can pick up a good weight, and convertes model to `SavedModel`/`h5` format by this command, it will add the Softmax layer in the last and cull the optimizer:

```bash
$ python tools/converter.py --model PATH/TO/MODEL --output PATH/TO/OUTPUT
```

And now `Tensorflow lite` also can convert this model, that means you can deploy it to Android, iOS etc.