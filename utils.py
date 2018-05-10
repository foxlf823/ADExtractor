from torch.utils.data import Dataset
import torch
from options import opt
import numpy as np
import logging
from tqdm import tqdm
import os
import shutil

# lowercased, number to 0, punctuation to #
ENG_PUNC = set(['`','~','!','@','#','$','%','&','*','(',')','-','_','+','=','{',
                '}','|','[',']','\\',':',';','\'','"','<','>',',','.','?','/'])

DIGIT = set(['0','1','2','3','4','5','6','7','8','9'])
def normalizeWord(word, cased=False):
    newword = ''
    for ch in word:
        if ch in DIGIT:
            newword = newword + '0'
        elif ch in ENG_PUNC:
            newword = newword + '#'
        else:
            newword = newword + ch

    if not cased:
        newword = newword.lower()
    return newword




def getRelatonInstance(tokens, entities, relations, word_vocab, relation_vocab, position_vocab1, position_vocab2):

    X = []
    Y = []

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i]

        for index, relation in doc_relation.iterrows():

            # find entity mention
            entity1 = doc_entity[(doc_entity['id']==relation['entity1_id'])].iloc[0]
            entity2 = doc_entity[(doc_entity['id'] == relation['entity2_id'])].iloc[0]
            # find all sentences between entity1 and entity2
            former = entity1 if entity1['start']<entity2['start'] else entity2
            latter = entity2 if entity1['start']<entity2['start'] else entity1
            context_token = doc_token[(doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
            words = []
            positions1 = []
            positions2 = []
            i = 0
            former_head = -1
            latter_head = -1
            for _, token in context_token.iterrows():
                if token['start'] >= former['start'] and token['end'] <= former['end']:
                    if former_head < i:
                        former_head = i
                if token['start'] >= latter['start'] and token['end'] <= latter['end']:
                    if latter_head < i:
                        latter_head = i

                i += 1

            if former_head == -1: # due to tokenization error, e.g., 10_197, hyper-CVAD-based vs hyper-CVAD
                logging.debug('former_head not found, entity {} {} {}'.format(former['id'], former['start'], former['text']))
                continue
            if latter_head == -1:
                logging.debug('latter_head not found, entity {} {} {}'.format(latter['id'], latter['start'], latter['text']))
                continue


            i = 0
            for _, token in context_token.iterrows():

                word = normalizeWord(token['text'])
                words.append(word_vocab.lookup(word))

                positions1.append(position_vocab1.lookup(former_head-i))
                positions2.append(position_vocab2.lookup(latter_head-i))

                i += 1





            X.append({'tokens': words, 'positions1': positions1, 'positions2':positions2})
            Y.append(relation_vocab.lookup(relation['type']))


    return X, Y






class RelationDataset(Dataset):

    def __init__(self, X, Y, max_seq_len):
        self.X = X
        self.Y = Y

        if max_seq_len > 0:
            self.set_max_seq_len(max_seq_len)
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def set_max_seq_len(self, max_seq_len):
        for x in self.X:
            x['tokens'] = x['tokens'][:max_seq_len]
            x['positions1'] = x['positions1'][:max_seq_len]
            x['positions2'] = x['positions2'][:max_seq_len]
        self.max_seq_len = max_seq_len






def my_collate(batch):
    x, y = zip(*batch)
    # extract input indices
    tokens = [s['tokens'] for s in x]
    positions1 = [s['positions1'] for s in x]
    positions2 = [s['positions2'] for s in x]

    lengths = [len(row) for row in tokens]
    max_len = max(lengths)

    tokens = pad_sequence(tokens, max_len, opt.pad_idx)
    positions1 = pad_sequence(positions1, max_len, opt.pad_idx)
    positions2 = pad_sequence(positions2, max_len, opt.pad_idx)
    y = torch.LongTensor(y).view(-1)
    if torch.cuda.is_available():
        y = y.cuda()
    return (tokens, positions1, positions2, y)


def pad_sequence(x, max_len, pad_idx):
    # pad to meet the need of PrimaryCapsule
    max_len = max(max_len, 5)

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(pad_idx)
    for i, row in enumerate(x):
        assert pad_idx not in row, 'EOS in sequence {}'.format(row)
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)
    if torch.cuda.is_available():
        padded_x.cuda()
    return padded_x







