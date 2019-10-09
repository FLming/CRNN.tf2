# CRNN论文实现 - tensorflow 2.0

这个仓库是CRNN的tensorflow2.0版本的再实现。

## 数据准备

为了训练CRNN网络，你需要准备一个映射文件，图片数据和其对应的标签。在example_data文件夹下有示例。

### [映射文件](./example_data/table.txt)

格式请看文件。这个文件里包含了需要识别的字符和blank字符（放在文件的开头和结尾都行，但是我发现tensorflow的tf.nn.ctc_greedy_decoder函数好像不能让我们自行决定blank标签是几，好像默认在最后，所以想要使用这个函数的话，需要放在最后，不过我也写了一个函数自行解码），目前blank字符的内容可以随意填写。

### 图片数据

我认为受网络本身限制，训练时的图片最好都是同一个宽度，换句话说图片中的字符在resize之后宽度应该差不多比较好。图片的高度必须是32，宽度要大于16。这两个限制想改需要改CNN网络的结构。

### [标签数据](./example_data/annotation.txt)

标签文件的格式如下:
```
[图片相对位置] [对应标签]
```

你可以运行一下dataset.py确保你的数据没有问题。

## 训练

```bash
python train.py -p ../data/train.txt -t ../data/table.txt 
```

在tensorboard中查看训练情况

```bash
tensorboard --logdir=tensorboard/
```

## 测试

```bash
python eval.py -p ../data/val.txt -t ../data/table.txt --checkpoint ckpt/2019-10-08-15-02-28/
```