import os

import numpy as np
import tensorflow as tf


class OCRDataLoader():
    def __init__(self, 
                 annotation_path, 
                 image_height, 
                 image_width, 
                 table_path, 
                 blank_index=0, 
                 batch_size=1, 
                 shuffle=False, 
                 repeat=1):
        
        imgpaths, labels = self.read_imagepaths_and_labels(annotation_path)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.size = len(imgpaths)

        file_init = tf.lookup.TextFileInitializer(table_path, 
                                                  tf.string, 
                                                  tf.lookup.TextFileIndex.WHOLE_LINE,
                                                  tf.int64,
                                                  tf.lookup.TextFileIndex.LINE_NUMBER)
        # default value for blank label
        self.table = tf.lookup.StaticHashTable(initializer=file_init, default_value=blank_index)

        dataset = tf.data.Dataset.from_tensor_slices((imgpaths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.size)
        dataset = dataset.map(self._decode_and_resize)
        # Pay attention to the location of the batch function.
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._convert_label)
        dataset = dataset.repeat(repeat)

        self.dataset = dataset

    def read_imagepaths_and_labels(self, annotation_path):
        """Read txt file to get image paths and labels."""

        imgpaths = []
        labels = []
        for annpath in annotation_path.split(','):
            annotation_folder = os.path.dirname(annpath)
            with open(annpath) as f:
                content = np.array([line.strip().split() for line in f.readlines()])
                imgpaths_local = content[:, 0]
                imgpaths_local = [os.path.join(annotation_folder, line.lstrip("/")) for line in imgpaths_local]
                labels_local = content[:, 1]
                imgpaths.extend(imgpaths_local)
                labels.extend(labels_local)

        return imgpaths, labels

    def _decode_and_resize(self, filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image_gray = tf.image.rgb_to_grayscale(image_decoded)
        image_resized = tf.image.resize(image_gray, [self.image_height, self.image_width]) / 255.0
        return image_resized, label

    def _convert_label(self, image, label):
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.table.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)
        return image, sparse_label

    def __call__(self):
        """Return tf.data.Dataset."""
        return self.dataset

    def __len__(self):
        return self.size


def map_to_chars(inputs, table, blank_index=0, merge_repeated=False):
    """Map to chars.
    
    Args:
        inputs: list of char ids.
        table: char map.
        blank_index: the index of blank.
        merge_repeated: True, Only if ctc_greedy_decoder is not used.

    Returns:
        lines: list of string.    
    """
    lines = []
    for line in inputs:
        text = ""
        previous_char = -1
        for char_index in line:
            if merge_repeated:
                if char_index == previous_char:
                    continue
            previous_char = char_index
            if char_index == blank_index:
                continue
            text += table[char_index]            
        lines.append(text)
    return lines


if __name__ == "__main__":
    dataloader = OCRDataLoader("example_data/val.txt", 32, 100, table_path="example_data/table.txt", shuffle=True, batch_size=2)
    print("Total have {} data".format(len(dataloader)))
    for image, label in dataloader().take(2):
        print("The image's shape: {}, label's dense shape is {}.".format(image.shape, label.dense_shape))