import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-traindata', type=str, default="/Users/feili/Desktop/umass/MADE/MADE-1.0_debug")
parser.add_argument('-testdata', type=str, default="/Users/feili/Desktop/umass/MADE/made_test_data_debug")
parser.add_argument('-batch', type=int, default=4)
parser.add_argument('-iter', type=int, default=1)
parser.add_argument('-display', type=int, default=50)
parser.add_argument('-whattodo', type=int, default=2, help='1-preprocess, 2-train, 3-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-emb', type=str, default="/Users/feili/project/emb_100_for_debug.txt")
parser.add_argument('-output', type=str, default="./output")
parser.add_argument('-max_seq_len', type=int, default=0) # set to <=0 to not truncate
parser.add_argument('-batch_size', type=int, default=2)
parser.add_argument('-max_epoch', type=int, default=30)

opt = parser.parse_args()