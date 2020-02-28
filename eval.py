import sys
import time
import argparse

import tensorflow as tf

from model import crnn
from dataset import read_img_paths_and_labels, map_to_chars

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotation_paths", type=str, required=True, 
                    nargs="+", help="The paths of annnotation file.")
parser.add_argument("-f", "--parse_funcs", type=str, required=True,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-k", "--keep_ratio", action="store_true",
                    help="Whether keep the ratio.")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-c", "--checkpoint", type=str, required=True, 
                    help="The checkpoint path.")
args = parser.parse_args()

num_invalid = 0

def load_data():
    img_paths, labels = read_img_paths_and_labels(
        args.annotation_paths, 
        args.parse_funcs)
    with open(args.table_path) as f:
        inv_table = [char.strip() for char in f]
    num_classes = len(inv_table)
    blank_index = num_classes - 1
    print("Num of eval samples: {}".format(len(img_paths)))
    print("Num of classes: {}".format(num_classes))
    print("Blank index is {}".format(blank_index))
    return img_paths, labels, inv_table, num_classes, blank_index

def read_image(path):
    img = tf.io.read_file(path)
    try:
        img = tf.io.decode_jpeg(img, channels=1)
    except Exception:
        print("Invalid image: {}".format(path))
        global num_invalid
        num_invalid += 1
        return tf.zeros((32, args.image_width, 1))
    img = tf.image.convert_image_dtype(img, tf.float32)
    if args.keep_ratio:
        width = round(32 * img.shape[1] / img.shape[0])
    else: 
        width = args.image_width
    img = tf.image.resize(img, (32, width))
    return img

def eval_one_step(model, x):
    logits = model(x, training=False)
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    return decoded

def eval(model, img_paths, labels, inv_table, blank_index):
    cnt = 0
    total_cnt = 0
    batch_size = args.batch_size if not args.keep_ratio else 1
    steps = round(len(img_paths) / batch_size) + 1
    for i in range(steps):
        start = time.perf_counter()
        x = img_paths[i * batch_size:(i + 1) * batch_size]
        if len(x) == 0:
            continue
        total_cnt += len(x)
        x = list(map(read_image, x))
        x = tf.stack(x)
        y = labels[i * batch_size:(i + 1) * batch_size]

        decoded = eval_one_step(model, x)
        decoded = tf.sparse.to_dense(decoded[0], 
                                     default_value=blank_index).numpy()
        decoded = map_to_chars(decoded, inv_table, blank_index=blank_index)
        for y_pred, y_true in zip(decoded, y):
            if y_pred == y_true:
                cnt += 1
        end = time.perf_counter()
        output = '{:.0%} Total: {}, Right: {}, S: {:.2f} images/s, Acc: {:.2%}'
        print(output.format((i + 1) / steps, total_cnt - num_invalid, cnt, 
                            batch_size / (end - start), 
                            cnt / (total_cnt - num_invalid)), end='')
        if i < steps - 1:
            print(end='\r')
    print()

if __name__ == "__main__":
    img_paths, labels, inv_table, num_classes, blank_index = load_data()
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(f"Start at {localtime}")

    model = crnn(num_classes)
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(
            tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        sys.exit()

    print("Evaluating...")
    eval(model, img_paths, labels, inv_table, blank_index)