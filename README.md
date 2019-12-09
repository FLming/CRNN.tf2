# Convolutional Recurrent Neural Network - tensorflow 2.0

This repo is a re-implement of [CRNN](http://arxiv.org/abs/1507.05717), [authors repo](https://github.com/bgshih/crnn).

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


## Train

```bash
python train.py -ta /PATH/TO/TXT -va /PATH/TO/TXT -t /PATH/TO/TABLE
```

For other parameters please check the train.py

The training process can be viewed using tensorboard

```bash
tensorboard --logdir=tensorboard/
```

![tensorboard](example/images/tensorboard.png)

## Eval

```bash
python eval.py -p /PATH/TO/TXT -t /PATH/TO/TABLE --checkpoint /PATH/TO/CHECKPOINT
```

For other parameters please check the eval.py

## Demo inference

```bash
python demo.py -i example/images/1_Paintbrushes_55044.jpg -t example/table.txt --checkpoint example/mjsynth/
```

then, You will see output:
```
[Beam search] prediction: Paintbrushes, log probabilities: -0.0010947763221338391
[Greedy] prediction: Paintbrushes, neg sum logits: -267.3738708496094
```

## Tensorflow serving

Please refer to the official website for [installation](https://www.tensorflow.org/tfx/serving/setup).

1. First you should pick a good model.
2. convert checkpoint to SavedModel
3. Just run tensorflow serving by 
```bash
tensorflow_model_server --rest_api_port=8501 --model_name=CRNN --model_base_path="/path/to/SavedModel/"
```