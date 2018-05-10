import numpy as np
import torch
import torch.nn as nn
import sortedcontainers
from options import opt
import logging

class Vocab:

    def __init__(self, alphabet_from_dataset, pretrained_file, emb_size):
        # add UNK with its index 0
        self.unk_tok = '<unk>'
        self.unk_idx = 0
        self.vocab_size = 1
        self.v2wvocab = ['<unk>']
        self.w2vvocab = {'<unk>': 0}
        # add padding with its index 1
        self.pad_tok = '<pad>'
        self.pad_idx = opt.pad_idx
        self.vocab_size += 1
        self.v2wvocab.append('<pad>')
        self.w2vvocab['<pad>'] = self.pad_idx
        # build vocabulary
        self.vocab_size += len(alphabet_from_dataset)
        cnt = 2
        for alpha in alphabet_from_dataset:
            self.v2wvocab.append(alpha)
            self.w2vvocab[alpha] = cnt
            cnt += 1
        # initialize embeddings
        if pretrained_file:
            ct_word_in_pretrained = 0
            with open(pretrained_file, 'r') as inf:
                parts = inf.readline().split()
                self.emb_size = int(parts[1]) # use pretrained embedding size
                self.embeddings = np.random.uniform(-0.01, 0.01, size=(self.vocab_size, self.emb_size))

                for line in inf.readlines():
                    parts = line.rstrip().split(' ')
                    word = parts[0] # not norm, use original word in the pretrained
                    if word == self.unk_tok or word == self.pad_tok:
                        continue # don't use the pretrained values for unk and pad
                    else:
                        if word in self.w2vvocab: # if we can, use the pretrained value
                            vector = [float(x) for x in parts[-self.emb_size:]]
                            self.embeddings[self.w2vvocab[word]] = vector
                            ct_word_in_pretrained += 1


            logging.info("{} in vocab matched {}".format(100.0*ct_word_in_pretrained/self.vocab_size, pretrained_file))

        else:
            self.emb_size = emb_size
            self.embeddings = np.random.uniform(-0.01, 0.01, size=(self.vocab_size, self.emb_size))

        # normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1,1)
        # zero pad emb
        self.embeddings[self.pad_idx] = 0

    def lookup(self, alpha):

        if alpha in self.w2vvocab:
            return self.w2vvocab[alpha]
        return self.unk_idx

    def lookup_id2str(self, id):
        if id<0 or id>=self.vocab_size:
            raise RuntimeError("{}: id out of range".format(self.__class__.__name__))
        return self.v2wvocab[id]






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


