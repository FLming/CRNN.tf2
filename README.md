# Convolutional Recurrent Neural Network - tensorflow 2.0

[中文](./README-zh.md)

This repo is a re-implement of [CRNN](http://arxiv.org/abs/1507.05717), [authors repo](https://github.com/bgshih/crnn). Thanks to the community and researchers for bringing technological innovation.

## Data prepare

In order to train this network, you should prepare a lookup table, some image and label data. Example data both on example_data folder.

### [Lookup table](./example_data/table.txt)

See example. The file contains all characters and blank label(in the last or any place both ok, but I find tf.nn.ctc_greedy_decoder set it to last and can't change it.) Currently, you can write any word for blank.

### Image data

In my opinion, images of the same width(>=16) will be better. By the way, the inputs of image height must equals 32 because of CNN construction. If you want to change it, you have to change the construction of CNN.

### [Lable data](./example_data/annotation.txt)

The format of annotation file is:
```
[Relative position of the image file] [lable]
```

You can run dataset.py to make sure your data prepare is ok.


## Train

```bash
python main.py --mode train --annotation_path XX.txt --table_path XX.txt 
```

See in tensorboard

```bash
tensorboard --logdir=tensorboard/
```

## Test

```bash
python main.py --mode test --annotation_path data/train.txt --table_path data/table.txt --checkpoint_path ckpt/XX
```