import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=4, help='1-preprocess, 2-pretrain, 3-train, other-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-gpu', type=int, default=0)

# directory
parser.add_argument('-traindata', type=str, default="/Users/feili/Desktop/umass/MADE/MADE-1.0_debug")
parser.add_argument('-testdata', type=str, default="/Users/feili/Desktop/umass/MADE/made_test_data_debug")
# parser.add_argument('-traindata', type=str, default="/Users/feili/Desktop/umass/MADE/MADE-1.0")
# parser.add_argument('-testdata', type=str, default="/Users/feili/Desktop/umass/MADE/made_test_data")
parser.add_argument('-output', type=str, default="./output")
parser.add_argument('-emb', type=str, default="/Users/feili/project/emb_100_for_debug.txt")
#parser.add_argument('-emb', type=str, default="/Users/feili/project/man/data/w2v/word2vec.txt")
# parser.add_argument('-emb', type=str, default="/Users/feili/project/abd_naacl_emb/pubmed+wiki+pitts-nopunct-lower-cbow-n10.bin")
parser.add_argument('-pretrain', type=str, default="./pretrain")

# preprocessing
parser.add_argument('-max_seq_len', type=int, default=0) # set to <=0 to not truncate
parser.add_argument('-full_data', action='store_true', default=False)
parser.add_argument('-pad_idx', default=1, type=int)
parser.add_argument('-sent_window', default=3, type=int)

# training
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-dropout', type=float, default=0.4)
# training strategy, all means use all instances, balance means each class use one in each batch
# no-unk means don't use unknown instance
parser.add_argument('-strategy', default='all', help='all, balance, no-unk, part-unk')
parser.add_argument('-unk_ratio', type=float, default=0.03) # only work when 'part-unk'

# hyper-parameter
parser.add_argument('-position_emb_size', default=10, type=int)
parser.add_argument('-word_emb_size', default=50, type=int)
parser.add_argument('-relation_emb_size', default=10, type=int)

# feature extractor
parser.add_argument('-model', default='cnn', help='cnn, lstm, other')
parser.add_argument('-shared_hidden_size', type=int, default=256)
parser.add_argument('-F_layers', type=int, default=1)
parser.add_argument('-kernel_num', type=int, default=200)
parser.add_argument('-kernel_sizes', type=int, nargs='+', default=[3,4,5])
parser.add_argument('-model_bn', action='store_true', default=False)

# high-level model
parser.add_argument('-model_high', default='capsule', help='capsule, mlp, capsule_em')
parser.add_argument('-dim_enlarge_rate', type=int, default=2)
parser.add_argument('-init_dim_cap', type=int, default=4)
parser.add_argument('-model_high_bn', action='store_true', default=False)




opt = parser.parse_args()

opt.max_kernel_size = max(opt.kernel_sizes)