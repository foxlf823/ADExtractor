

from torch.utils.data import Dataset
import torch
from options import opt
import numpy as np
import logging
from tqdm import tqdm
import os
import shutil
import math

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

# relation constraints
# 'do': set(['Drug Dose', 'Dose Dose']),
# 'fr': set(['Drug Frequency', 'Frequency Frequency']),
# 'manner/route': set(['Drug Route', 'Route Route']),
# 'Drug_By Patient': set(['Drug By Patient']),
# 'severity_type': set(['Indication Severity', 'ADE Severity', 'SSLIF Severity', 'Severity Severity']),
# 'adverse': set(['Drug ADE', 'SSLIF ADE', 'ADE ADE']),
# 'reason': set(['Drug Indication', 'Indication Indication']),
# 'Drug_By Physician': set(['Drug By Physician']),
# 'du': set(['Duration Duration', 'Drug Duration'])

def relationConstraint(entity1, entity2): # determine whether the constraint are satisfied, non-directional
    type1, type2 = entity1['type'], entity2['type']

    if (type1 == 'Drug' and type2 == 'Dose') or (type1 == 'Dose' and type2 == 'Drug') or (type1 == 'Dose' and type2 == 'Dose'):
        return True
    elif (type1 == 'Drug' and type2 == 'Frequency') or (type1 == 'Frequency' and type2 == 'Drug') or (type1 == 'Frequency' and type2 == 'Frequency'):
        return True
    elif (type1 == 'Drug' and type2 == 'Route') or (type1 == 'Route' and type2 == 'Drug') or (type1 == 'Route' and type2 == 'Route'):
        return True
    elif (type1 == 'Drug By' and type2 == 'Patient') or (type1 == 'Patient' and type2 == 'Drug By'):
        return True
    elif (type1 == 'Indication' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'Indication') or \
                (type1 == 'ADE' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'ADE') or \
                (type1 == 'SSLIF' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'SSLIF') \
            or (type1 == 'Severity' and type2 == 'Severity'):
        return True
    elif (type1 == 'Drug' and type2 == 'ADE') or (type1 == 'ADE' and type2 == 'Drug') or\
                (type1 == 'SSLIF' and type2 == 'ADE') or (type1 == 'ADE' and type2 == 'SSLIF') \
            or (type1 == 'ADE' and type2 == 'ADE'):
        return True
    elif (type1 == 'Drug' and type2 == 'Indication') or (type1 == 'Indication' and type2 == 'Drug') or (type1 == 'Indication' and type2 == 'Indication'):
        return True
    elif (type1 == 'Drug By' and type2 == 'Physician') or (type1 == 'Physician' and type2 == 'Drug By'):
        return True
    elif (type1 == 'Drug' and type2 == 'Duration') or (type1 == 'Duration' and type2 == 'Drug') or (type1 == 'Duration' and type2 == 'Duration'):
        return True
    else:
        return False

# def relationConstraint(relation_type, entity1, entity2):
#     if relation_type=='do':
#         if entity1['type']== 'Drug' and entity2['type']=='Dose':
#             return 1
#         elif entity1['type']== 'Dose' and entity2['type']=='Dose':
#             return 2
#
#     elif relation_type=='fr':
#         pass
#     elif relation_type=='manner/route':
#         pass
#     elif relation_type=='Drug_By Patient':
#         pass
#     elif relation_type=='severity_type':
#         pass
#     elif relation_type=='adverse':
#         pass
#     elif relation_type=='reason':
#         pass
#     elif relation_type=='Drug_By Physician':
#         pass
#     elif relation_type=='du':
#         pass
#     else:
#         raise RuntimeError("unknown relation type")


# enumerate all entity pairs
def getRelationInstance1(tokens, entities, relations, word_vocab, relation_vocab, position_vocab1, position_vocab2):
    X = []
    Y = []
    cnt_neg = 0

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i] # entity are sorted by start offset

        row_num = doc_entity.shape[0]

        # cnt1 = 0
        # cnt2 = 0
        # cnt3 = 0
        for latter_idx in range(row_num):

            for former_idx in range(row_num):

                if former_idx < latter_idx:
                    # cnt1 += 1
                    former = doc_entity.loc[former_idx]
                    latter = doc_entity.loc[latter_idx]

                    if math.fabs(latter['sent_idx']-former['sent_idx']) >= opt.sent_window:
                        continue

                    # cnt2 += 1

                    if relationConstraint(former, latter) == False:
                        continue

                    # cnt3 += 1
                    gold_relations = doc_relation[
                        (
                                ((doc_relation['entity1_id'] == former['id']) & (
                                            doc_relation['entity2_id'] == latter['id']))
                                |
                                ((doc_relation['entity1_id'] == latter['id']) & (
                                            doc_relation['entity2_id'] == former['id']))
                        )
                    ]
                    if gold_relations.shape[0] > 1:
                        raise RuntimeError("the same entity pair has more than one relations")



                    context_token = doc_token[
                        (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
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

                    if former_head == -1:  # due to tokenization error, e.g., 10_197, hyper-CVAD-based vs hyper-CVAD
                        logging.debug('former_head not found, entity {} {} {}'.format(former['id'], former['start'],
                                                                                      former['text']))
                        continue
                    if latter_head == -1:
                        logging.debug('latter_head not found, entity {} {} {}'.format(latter['id'], latter['start'],
                                                                                      latter['text']))
                        continue


                    i = 0
                    for _, token in context_token.iterrows():
                        word = normalizeWord(token['text'])
                        words.append(word_vocab.lookup(word))

                        positions1.append(position_vocab1.lookup(former_head - i))
                        positions2.append(position_vocab2.lookup(latter_head - i))

                        i += 1

                    X.append({'tokens': words, 'positions1': positions1, 'positions2': positions2})
                    if gold_relations.shape[0] == 0:
                        Y.append(relation_vocab.lookup('<unk>'))
                        cnt_neg += 1
                    else:
                        Y.append(relation_vocab.lookup(gold_relations.iloc[0]['type']))







        #print(cnt1, cnt2, cnt3)


    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y



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






# def my_collate(batch):
#     x, y = zip(*batch)
#     # extract input indices
#     tokens = [s['tokens'] for s in x]
#     positions1 = [s['positions1'] for s in x]
#     positions2 = [s['positions2'] for s in x]
#
#     lengths = [len(row) for row in tokens]
#     max_len = max(lengths)
#
#     tokens = pad_sequence(tokens, max_len, opt.pad_idx)
#     positions1 = pad_sequence(positions1, max_len, opt.pad_idx)
#     positions2 = pad_sequence(positions2, max_len, opt.pad_idx)
#     y = torch.LongTensor(y).view(-1)
#     if torch.cuda.is_available():
#         y = y.cuda()
#     return (tokens, positions1, positions2, y)
#
#
# def pad_sequence(x, max_len, pad_idx):
#     # pad to meet the need of PrimaryCapsule
#     max_len = max(max_len, 5)
#
#     padded_x = np.zeros((len(x), max_len), dtype=np.int)
#     padded_x.fill(pad_idx)
#     for i, row in enumerate(x):
#         assert pad_idx not in row, 'EOS in sequence {}'.format(row)
#         padded_x[i][:len(row)] = row
#     padded_x = torch.LongTensor(padded_x)
#     if torch.cuda.is_available():
#         padded_x = padded_x.cuda()
#     return padded_x

def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    x, y = zip(*batch)
    # extract input indices
    tokens = [s['tokens'] for s in x]
    positions1 = [s['positions1'] for s in x]
    positions2 = [s['positions2'] for s in x]

    (tokens, positions1, positions2, lengths), y = pad(tokens, positions1, positions2, y, opt.pad_idx, sort)
    if torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        positions1 = positions1.cuda(opt.gpu)
        positions2 = positions2.cuda(opt.gpu)
        lengths = lengths.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return tokens, positions1, positions2, lengths, y



def pad(tokens, positions1, positions2, y, eos_idx, sort):
    lengths = [len(row) for row in tokens]
    max_len = max(lengths)
    # if using CNN, pad to at least the largest kernel size
    if opt.model.lower() == 'cnn':
        max_len = max(max_len, opt.max_kernel_size)
    # pad sequences
    tokens = pad_sequence(tokens, max_len, eos_idx)
    positions1 = pad_sequence(positions1, max_len, eos_idx)
    positions2 = pad_sequence(positions2, max_len, eos_idx)
    lengths = torch.LongTensor(lengths)
    y = torch.LongTensor(y).view(-1)
    if sort:
        # sort by length
        sort_len, sort_idx = lengths.sort(0, descending=True)
        tokens = tokens.index_select(0, sort_idx)
        positions1 = positions1.index_select(0, sort_idx)
        positions2 = positions2.index_select(0, sort_idx)
        y = y.index_select(0, sort_idx)
        return (tokens, positions1, positions2, sort_len), y
    else:
        return (tokens, positions1, positions2, lengths), y


def pad_sequence(x, max_len, eos_idx):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, 'EOS in sequence {}'.format(row)
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)

    return padded_x




