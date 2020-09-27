import re
import itertools
from pathlib import Path

import tensorflow as tf


class UnsupportedFormatError(Exception):
    """Error class for unsupported format"""


def read_annotation(path: Path):
    """Read an annotation file to get image paths and labels."""
    with path.open() as f:
        line = f.readline().strip()
        if re.fullmatch(r'.*/*\d+_.+_(\d+)\.\w+ \1', line):
            annotation_format = 'MJSynth'
            content = [l.strip().split() for l in itertools.chain([line], f)]
            img_paths, labels = zip(*content)
            labels = [path.split('_')[1] for path in img_paths]
        elif re.fullmatch(r'.*/*word_\d\.\w+, ".+"', line):
            annotation_format = 'ICDAR2013/ICDAR2015'
            content = [l.strip().split(',') for l in itertools.chain([line], f)]
            img_paths, labels = zip(*content)
            labels = [label.strip(' "') for label in labels]
        elif re.fullmatch(r'.+\.\w+ .+', line):
            annotation_format = 'Simple'
            content = [l.strip().split() for l in itertools.chain([line], f)]
            img_paths, labels = zip(*content)
        else:
            raise UnsupportedFormatError('Unsupported annotation format')
    img_paths = [str(path.parent / img_path) for img_path in img_paths]
    print(f'Annotation path: {path}, format: {annotation_format}')
    return img_paths, labels


def read_annotations(paths):
    """Read annotation files to get image paths and labels."""
    img_paths = []
    labels = []
    for path in paths:
        part_img_paths, part_labels = read_annotation(Path(path))
        img_paths.extend(part_img_paths)
        labels.extend(part_labels)
    return img_paths, labels


class DatasetBuilder():

    def __init__(self, table_path, img_channels, max_img_width=300, 
                 ignore_case=False, img_width=None, img_height=32):
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, 
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), -1)
        self.img_height = img_height
        self.img_channels = img_channels
        self.ignore_case = ignore_case
        if img_width is None:
            self.max_img_width = max_img_width
            self.preserve_aspect_ratio = True
        else:
            self.img_width = img_width
            self.preserve_aspect_ratio = False

    @property
    def num_classes(self):
        return self.table.size()

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_channels)
        if self.preserve_aspect_ratio:
            img_shape = tf.shape(img)
            scale_factor = self.img_height / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_width
        img = tf.image.resize(img, (self.img_height, img_width)) / 255.0
        return img, label

    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] < self.max_img_width

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens
        
    def build(self, ann_paths, batch_size, is_training):
        """
        build dataset, it will auto detect each annotation file's format.
        """
        img_paths, labels = read_annotations(ann_paths)
        if self.ignore_case:
            labels = list(map(str.lower, labels))
        print(f'Number of samples: {len(img_paths)}')
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Ignore the errors e.g. decode error or invalid data.
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if self.preserve_aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)      
            ds = ds.padded_batch(batch_size, drop_remainder=is_training)
        else:
            ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.map(self._tokenize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds