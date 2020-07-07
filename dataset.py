import os
import re

import tensorflow as tf


class UnsupportedFormatError(Exception):
    """Error class for unsupported format"""


def read_annotation(path):
    """Read an annotation file to get image paths and labels."""
    print(f'Annotation path: {path}, format: ', end='')
    with open(path) as f:
        line = f.readline().strip()
        if re.fullmatch(r'.*/*\d+_.+_(\d+)\.\w+ \1', line):
            print('MJSynth')
            content = [l.strip().split() for l in f.readlines() + [line]]
            img_paths, labels = zip(*content)
            labels = [path.split('_')[1] for path in img_paths]
        elif re.fullmatch(r'.*/*word_\d\.\w+, ".+"', line):
            print('ICDAR2013')
            content = [l.strip().split(',') for l in f.readlines() + [line]]
            img_paths, labels = zip(*content)
            labels = [label.strip(' "') for label in labels]
        elif re.fullmatch(r'.+\.\w+ .+', line):
            print('[image path] label')
            content = [l.strip().split() for l in f.readlines() + [line]]
            img_paths, labels = zip(*content)
        else:
            raise UnsupportedFormatError('Unsupported annotation format')
    dirname = os.path.dirname(path)
    img_paths = [os.path.join(dirname, img_path) for img_path in img_paths]
    return img_paths, labels


def read_annotations(paths):
    """Read annotation files to get image paths and labels."""
    img_paths = []
    labels = []
    for path in paths:
        part_img_paths, part_labels = read_annotation(path)
        img_paths.extend(part_img_paths)
        labels.extend(part_labels)
    return img_paths, labels


class DatasetBuilder():
    def __init__(self, table_path, img_width, img_channels, ignore_case=False):
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, 
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), -1)
        self.img_width = img_width
        self.img_channels = img_channels
        self.ignore_case = ignore_case
        self.num_classes = self.table.size()

    def decode_and_resize(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (32, self.img_width))
        return img, label

    def tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens
        
    def build(self, ann_paths, shuffle, batch_size):
        """
        build dataset, it will auto detect each annotation file's format.
        """
        img_paths, labels = read_annotations(ann_paths)
        if self.ignore_case:
            labels = [label.lower() for label in labels]
        size = len(img_paths)
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self.decode_and_resize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Ignore the errors e.g. decode error or invalid data.
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.batch(batch_size)
        ds = ds.map(self.tokenize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds, size


class Decoder:
    def __init__(self, table, blank_index=-1, merge_repeated=True):
        """
        
        Args:
            table: list, char map
            blank_index: int(default: num_classes - 1), the index of blank 
        label.
            merge_repeated: bool
        """
        self.table = table
        if blank_index == -1:
            blank_index = len(table) - 1
        self.blank_index = blank_index
        self.merge_repeated = merge_repeated

    def map2string(self, inputs):
        strings = []
        for i in inputs:
            text = [self.table[char_index] for char_index in i 
                    if char_index != self.blank_index]
            strings.append(''.join(text))
        return strings

    def decode(self, inputs, from_pred=True, method='greedy'):
        if from_pred:
            logit_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            if method == 'greedy':
                decoded, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length,
                    merge_repeated=self.merge_repeated)
            elif method == 'beam_search':
                decoded, _ = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length)
            inputs = decoded[0]
        decoded = tf.sparse.to_dense(inputs, 
                                     default_value=self.blank_index).numpy()
        decoded = self.map2string(decoded)
        return decoded