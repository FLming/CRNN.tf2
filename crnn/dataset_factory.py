import re
from pathlib import Path

import tensorflow as tf


class UnsupportedFormatError(Exception):
    """Error class for unsupported format"""


def read_annotation(path: Path):
    """Read an annotation file to get image paths and labels."""
    print(f'Annotation path: {path}, format: ', end='')
    with path.open() as f:
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
            print('Simple')
            content = [l.strip().split() for l in f.readlines() + [line]]
            img_paths, labels = zip(*content)
        else:
            raise UnsupportedFormatError('Unsupported annotation format')
    img_paths = [str(path.parent / img_path) for img_path in img_paths]
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

    def __init__(self, table_path, img_width, img_channels, ignore_case=False, 
                 img_height=32):
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, 
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), -1)
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.ignore_case = ignore_case

    @property
    def num_classes(self):
        return self.table.size()

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        return img, label

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens
        
    def build(self, ann_paths, batch_size, shuffle):
        """
        build dataset, it will auto detect each annotation file's format.
        """
        img_paths, labels = read_annotations(ann_paths)
        if self.ignore_case:
            labels = list(map(str.lower, labels))
        print(f'Num of samples: {len(img_paths)}')
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        ds = ds.map(self._decode_img, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Ignore the errors e.g. decode error or invalid data.
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.batch(batch_size)
        ds = ds.map(self._tokenize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds