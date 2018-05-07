import numpy as np
import torch
import torch.nn as nn
import sortedcontainers
from options import opt


class WordVocab:
    def __init__(self, txt_file):
        with open(txt_file, 'r') as inf:
            parts = inf.readline().split()
            assert len(parts) == 2
            self.vocab_size, self.emb_size = int(parts[0]), int(parts[1])
            opt.vocab_size = self.vocab_size
            opt.emb_size = self.emb_size
            # add an UNK token
            self.unk_tok = '<unk>'
            self.unk_idx = 0
            self.vocab_size += 1
            self.v2wvocab = ['<unk>']
            self.w2vvocab = {'<unk>': 0}
            self.embeddings = np.empty((self.vocab_size, self.emb_size), dtype=np.float)
            cnt = 1
            for line in inf.readlines():
                parts = line.rstrip().split(' ')
                word = parts[0]
                # add to vocab
                self.v2wvocab.append(word)
                self.w2vvocab[word] = cnt
                # load vector
                vector = [float(x) for x in parts[-self.emb_size:]]
                self.embeddings[cnt] = vector
                cnt += 1

        self.eos_tok = '</s>'
        opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
        # randomly initialize <unk> vector
        self.embeddings[self.unk_idx] = np.random.normal(0, 1, size=self.emb_size)
        # normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1,1)
        # zero </s>
        self.embeddings[self.eos_idx] = 0


    def init_embed_layer(self):
        word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.eos_idx)
        if not opt.random_emb:
            word_emb.weight.data = torch.from_numpy(self.embeddings).float()
        return word_emb

    def lookup(self, word):

        if word in self.w2vvocab:
            return self.w2vvocab[word]
        return self.unk_idx


            

class RelationVocab:

    def __init__(self, relations):

        relationName = sortedcontainers.SortedSet()
        for relation in relations:
            relationName.update(relation['type'].tolist())

        self.id2str = list(relationName)
        self.vocab_size = len(self.id2str)
        self.str2id = {}
        cnt = 0
        for str in self.id2str:
            self.str2id[str] = cnt
            cnt += 1
        # add an UNK relation
        self.unk = '<unk>'
        self.unk_idx = self.vocab_size
        self.id2str.append(self.unk)
        self.str2id[self.unk] = self.unk_idx
        self.vocab_size += 1

    def lookup(self, item):

        if item in self.str2id:
            return self.str2id[item]
        return self.unk_idx


