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

### Tips

About data pipline, you can change the dataset.py. For example only deal with digit data, then filt in read_imagepaths_and_labels function.

About data augment, you can use tf.image.xx api.

## Train

```bash
python train.py -p ../data/train.txt -t ../data/table.txt
```

See in tensorboard.

```bash
tensorboard --logdir=tensorboard/
```

## Eval

```bash
python eval.py -p ../data/val.txt -t ../data/table.txt --checkpoint ckpt/2019-10-08-15-02-28/
```

## Tensorflow serving

Please refer to the official website for [installation](https://www.tensorflow.org/tfx/serving/setup).

1. First you should pick a good model.
2. Run to_SavedModel.py get a SavedModel format model.
3. Just run tensorflow serving by 
```bash
tensorflow_model_server --rest_api_port=8501 --model_name=CRNN --model_base_path="/path/to/SavedModel/"
```

## Tensorflow lite

Still have problem. It seems not support RNN and None-shape input now. If you know how to convert, tell me thanks.

If you defined a fixed-shape input and remove BiLSTM, you can run to_tflite.py to get a tflite file.

## OpenVINO

Still have a problem to convert to openvino ir.