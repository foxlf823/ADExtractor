from torch.utils.data import Dataset
import torch
from options import opt
import numpy as np

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




def getRelatonInstance(tokens, entities, relations, word_vocab, relation_vocab):

    X = []
    Y = []

    for i, doc_relation in enumerate(relations):

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
            for word in context_token['text']:
                word = normalizeWord(word)
                words.append(word_vocab.lookup(word))
            X.append({'tokens': words})
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
        self.max_seq_len = max_seq_len





def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    x, y = zip(*batch)
    # extract input indices
    x = [s['tokens'] for s in x]
    x, y = pad(x, y, opt.eos_idx, sort)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    return (x, y)


def pad(x, y, eos_idx, sort):
    lengths = [len(row) for row in x]
    max_len = max(lengths)

    # pad sequences
    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, 'EOS in sequence {}'.format(row)
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)
    y = torch.LongTensor(y).view(-1)

    return padded_x, y





