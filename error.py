import logging
import cPickle as pickle
import os
from options import opt

logging.info("loading ... vocab")
word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

# One relation instance is composed of X (a pair of entities and their context), Y (relation label).
train_X = pickle.load(open(os.path.join(opt.pretrain, 'train_X.pkl'), 'rb'))
train_Y = pickle.load(open(os.path.join(opt.pretrain, 'train_Y.pkl'), 'rb'))
logging.info("total training instance {}".format(len(train_Y)))

for i, x in enumerate(train_X):
    y = train_Y[i]
    relation_str = relation_vocab.lookup_id2str(y)
    if relation_str == "adverse":
        sequence = ""
        for j, token_id in enumerate(x['tokens']):

            p1, p2 = position_vocab1.lookup_id2str(x['positions1'][j]), position_vocab2.lookup_id2str(x['positions2'][j])
            token = word_vocab.lookup_id2str(token_id)
            if p1 == 0 or p2 == 0:
                sequence += "["+token+"] "
            else:
                sequence += token + " "
        print(sequence)