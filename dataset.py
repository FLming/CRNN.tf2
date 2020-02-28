import os

import tensorflow as tf

class OCRDataLoader():
    def __init__(self, annotation_paths, parse_funcs, image_width, table_path,
                 batch_size=64, shuffle=False, repeat=1):
        img_paths, labels = read_img_paths_and_labels(
            annotation_paths, 
            parse_funcs)
        self.image_width = image_width
        self.batch_size = batch_size
        self.size = len(img_paths)

        with open(table_path) as f:
            self.inv_table = [char.strip() for char in f]
        self.num_classes = len(self.inv_table)
        self.blank_index = self.num_classes - 1

        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), self.blank_index)

        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=self.size)
        ds = ds.map(self._decode_and_resize)
        # Experimental function.
        # Ignore the errors e.g. decode error or invalid data.
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.batch(batch_size).map(self._convert_label).repeat(repeat)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = ds

    def _decode_and_resize(self, filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (32, self.image_width))
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

    def __len__(self):
        return self.size

    def __call__(self):
        """Return tf.data.Dataset."""
        return self.dataset

def parse_mjsynth(annotation_path):
    """Parse MjSynth dataset. format: XX_label_XX.jpg XX.
    URL: https://www.robots.ox.ac.uk/~vgg/data/text/
    """
    dirname = os.path.dirname(annotation_path)
    with open(annotation_path) as f:
        content = [line.strip().split() for line in f.readlines()]
    img_paths = [os.path.join(dirname, v[0]) for v in content]
    labels = [v[0].split("_")[1] for v in content]
    return img_paths, labels

def parse_example(annotation_path):
    """Parse example dataset. format: XX.jpg label"""
    dirname = os.path.dirname(annotation_path)
    with open(annotation_path) as f:
        content = [line.strip().split() for line in f.readlines()]
    img_paths = [os.path.join(dirname, v[0]) for v in content]
    labels = [v[1] for v in content]
    return img_paths, labels

def parse_icdar2013(annotation_path):
    dirname = os.path.dirname(annotation_path)
    with open(annotation_path) as f:
        content = [line.strip().split(",") for line in f.readlines()]
    img_paths = [os.path.join(dirname, v[0]) for v in content]
    labels = [v[1].strip(' "') for v in content]
    return img_paths, labels

parse_func_map = {"mjsynth": parse_mjsynth, "example": parse_example, 
                  "icdar2013": parse_icdar2013}

def read_img_paths_and_labels(annotation_paths, funcs):
    """Read annotation files to get image paths and labels."""
    img_paths = []
    labels = []
    for annotation_path, func in zip(annotation_paths, funcs):
        part_img_paths, part_labels = parse_func_map[func](annotation_path)
        img_paths.extend(part_img_paths)
        labels.extend(part_labels)
    return img_paths, labels

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

def map_and_count(decoded, Y, table, blank_index=0, merge_repeated=False):
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    decoded = map_to_chars(decoded, table, blank_index=blank_index, 
                           merge_repeated=merge_repeated)
    Y = tf.sparse.to_dense(Y, default_value=blank_index).numpy()
    Y = map_to_chars(Y, table, blank_index=blank_index, 
                     merge_repeated=merge_repeated)
    cnt = 0
    for y_pred, y in zip(decoded, Y):
        if y_pred == y:
            cnt += 1
    return cnt

if __name__ == "__main__":
    import time 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_paths", type=str, required=True, 
                        help="The paths of annnotation file.")
    parser.add_argument("-t", "--table_path", type=str, required=True, 
                        help="The path of table file.")
    args = parser.parse_args()

    def timeit(ds, steps):
        overall_start = time.perf_counter()
        it = iter(ds.take(steps + 1))
        next(it)

        start = time.perf_counter()
        for i, (images, labels) in enumerate(it):
            if i % 10 == 0:
                print('.', end='')
        print()
        end = time.perf_counter()

        duration = end - start
        print("{} batches: {} s".format(steps, duration))
        print("{:0.5f} Images/s".format(64 * steps / duration))
        print("Total time: {}s".format(end - overall_start))

    dl = OCRDataLoader(
        args.annotation_paths, 
        [parse_example], 
        100, 
        args.table_path,
        repeat=None)
    
    timeit(dl(), 64)
    for x, y in dl().take(1):
        print(tf.sparse.to_dense(y))