import os

import numpy as np
import tensorflow as tf


class OCRDataLoader():
    def __init__(self, 
                 annotation_paths, 
                 image_height, 
                 image_width, 
                 table_path, 
                 blank_index=0, 
                 batch_size=1, 
                 shuffle=False, 
                 repeat=1):
        
        img_paths, labels = self._read_img_paths_and_labels(annotation_paths)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.size = len(img_paths)

        file_init = tf.lookup.TextFileInitializer(
            table_path, 
            tf.string, 
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER)
        # Default value for blank label
        self.table = tf.lookup.StaticHashTable(
            initializer=file_init, 
            default_value=blank_index)

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.size)
        dataset = dataset.map(self._decode_and_resize)
        # Experimental function.
        # Ignore the errors e.g. decode error or invalid data.
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._convert_label)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    def _read_img_paths_and_labels(self, annotation_paths):
        """Read annotation files to get image paths and labels."""

        img_paths = []
        labels = []
        for annotation_path in annotation_paths.split(','):
            # If you use your own dataset, maybe you should change the 
            # parse code below.
            annotation_folder = os.path.dirname(annotation_path)
            with open(annotation_path) as f:
                content = np.array(
                    [line.strip().split() for line in f.readlines()])
            part_img_paths = content[:, 0]
            # Parse MjSynth dataset. format: XX_label_XX.jpg XX
            # URL: https://www.robots.ox.ac.uk/~vgg/data/text/            
            part_labels = [line.split("_")[1] for line in part_img_paths]
            # Parse example dataset. format: XX.jpg label
            # part_labels = content[:, 1]
            part_img_paths = [os.path.join(annotation_folder, line)
                              for line in part_img_paths]
            img_paths.extend(part_img_paths)
            labels.extend(part_labels)

        return img_paths, labels

    def _decode_and_resize(self, filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.image_height, self.image_width])
        return image, label

    def _convert_label(self, image, label):
        # According to official document, only dense tensor will run on GPU
        # or TPU, but I have tried convert label to dense tensor by `to_tensor`
        # and `row_lengths`, the speed of training step slower than sparse.
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
        merge_repeated: True, Only if tf decoder is not used.

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

def map_and_count(decoded, Y, mapper, blank_index=0, merge_repeated=False):
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    decoded = map_to_chars(decoded, mapper, blank_index=blank_index, 
                           merge_repeated=merge_repeated)
    Y = tf.sparse.to_dense(Y, default_value=blank_index).numpy()
    Y = map_to_chars(Y, mapper, blank_index=blank_index, 
                     merge_repeated=merge_repeated)
    cnt = 0
    for y_pred, y in zip(decoded, Y):
        if y_pred == y:
            cnt += 1
    return cnt

if __name__ == "__main__":
    import argparse
    import time

    import arg

    parser = argparse.ArgumentParser(parents=[arg.parser])
    parser.add_argument("-p", "--annotation_paths", type=str, 
                        help="The paths of annnotation file.")
    args = parser.parse_args()

    dataloader = OCRDataLoader(
        args.annotation_paths, 
        args.image_height, 
        args.image_width, 
        table_path=args.table_path, 
        shuffle=True, 
        batch_size=32)
    print("Total have {} data".format(len(dataloader)))
    print("Element spec is: {}".format(dataloader().element_spec))
    start_time = time.perf_counter()
    for i in range(2):
        for image, label in dataloader().take(1000):
            pass
    print("Execution time:", time.perf_counter() - start_time)