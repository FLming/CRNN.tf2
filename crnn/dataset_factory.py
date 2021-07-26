import os
import re

import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.data.TextLineDataset):
    def __init__(self, filename, **kwargs):
        self.dirname = os.path.dirname(filename)
        super().__init__(filename, **kwargs)

    def parse_func(self, line):
        raise NotImplementedError

    def parse_line(self, line):
        line = tf.strings.strip(line)
        img_relative_path, label = self.parse_func(line)
        img_path = tf.strings.join([self.dirname, os.sep, img_relative_path])
        return img_path, label


class SimpleDataset(Dataset):

    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path, label = splited_line[0], splited_line[1]
        return img_relative_path, label


class MJSynthDataset(Dataset):

    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path = splited_line[0]
        label = tf.strings.split(img_relative_path, sep='_')[1]
        return img_relative_path, label


class ICDARDataset(Dataset):

    def parse_func(self, line):
        splited_line = tf.strings.split(line, sep=',')
        img_relative_path, label = splited_line[0], splited_line[1]
        label = tf.strings.strip(label)
        label = tf.strings.regex_replace(label, r'"', '')
        return img_relative_path, label


class DatasetBuilder:

    def __init__(self, table_path, img_shape=(32, None, 3), max_img_width=300,
                 ignore_case=False):
        # map unknown label to 0
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), 0)
        self.img_shape = img_shape
        self.ignore_case = ignore_case
        if img_shape[1] is None:
            self.max_img_width = max_img_width
            self.preserve_aspect_ratio = True
        else:
            self.preserve_aspect_ratio = False

    @property
    def num_classes(self):
        return self.table.size()

    def _parse_annotation(self, path):
        with open(path) as f:
            line = f.readline().strip()
        if re.fullmatch(r'.*/*\d+_.+_(\d+)\.\w+ \1', line):
            return MJSynthDataset(path)
        elif re.fullmatch(r'.*/*word_\d\.\w+, ".+"', line):
            return ICDARDataset(path)
        elif re.fullmatch(r'.+\.\w+ .+', line):
            return SimpleDataset(path)
        else:
            raise ValueError('Unsupported annotation format')

    def _concatenate_ds(self, ann_paths):
        datasets = [self._parse_annotation(path) for path in ann_paths]
        concatenated_ds = datasets[0].map(datasets[0].parse_line)
        for ds in datasets[1:]:
            ds = ds.map(ds.parse_line)
            concatenated_ds = concatenated_ds.concatenate(ds)
        return concatenated_ds

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        if self.preserve_aspect_ratio:
            img_shape = tf.shape(img)
            scale_factor = self.img_shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_shape[1]
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, label

    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] < self.max_img_width

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens

    def __call__(self, ann_paths, batch_size, is_training):
        ds = self._concatenate_ds(ann_paths)
        if self.ignore_case:
            ds = ds.map(lambda x, y: (x, tf.strings.lower(y)))
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img, AUTOTUNE)
        if self.preserve_aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)
            ds = ds.padded_batch(batch_size, drop_remainder=is_training)
        else:
            ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds
