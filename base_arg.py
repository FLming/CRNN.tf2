import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-t", "--table_path", type=str, required=True, help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, help="Image width(>=16).")
parser.add_argument("--image_height", type=int, default=32, help="Image height(32). If you change, you should change the structure of CNN.")
parser.add_argument("--backbone",
                    type=str,
                    default="VGG", 
                    choices=("VGG", "ResNet"),
                    help="The backbone of CRNNs, available now is VGG and ResNet.")