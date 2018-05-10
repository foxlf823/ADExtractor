import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-traindata', type=str, default="/Users/feili/Desktop/umass/MADE/MADE-1.0_debug")
# parser.add_argument('-testdata', type=str, default="/Users/feili/Desktop/umass/MADE/made_test_data_debug")
parser.add_argument('-traindata', type=str, default="/Users/feili/Desktop/umass/MADE/MADE-1.0")
parser.add_argument('-testdata', type=str, default="/Users/feili/Desktop/umass/MADE/made_test_data")
parser.add_argument('-whattodo', type=int, default=3, help='1-preprocess, 2-train, other-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-random_seed', type=int, default=1)
#parser.add_argument('-emb', type=str, default="/Users/feili/project/emb_100_for_debug.txt")
parser.add_argument('-emb', type=str, default="/Users/feili/project/man/data/w2v/word2vec.txt")
parser.add_argument('-output', type=str, default="./output")
parser.add_argument('-max_seq_len', type=int, default=0) # set to <=0 to not truncate
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-full_data', action='store_true', default=False)
parser.add_argument('-position_emb_size', default=10, type=int)
parser.add_argument('-word_emb_size', default=50, type=int)
parser.add_argument('-relation_emb_size', default=10, type=int)
parser.add_argument('-pad_idx', default=1, type=int)
parser.add_argument('-model', default=1, type=int, help='1-capsule, other-baseline')


opt = parser.parse_args()