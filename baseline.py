import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, word_vocab, position1_vocab, position2_vocab, relation_vocab):
        super(CNN, self).__init__()

        self.word_emb = nn.Embedding(word_vocab.vocab_size, word_vocab.emb_size, padding_idx=word_vocab.pad_idx)
        self.word_emb.weight.data = torch.from_numpy(word_vocab.embeddings).float()

        self.position1_emb = nn.Embedding(position1_vocab.vocab_size, position1_vocab.emb_size,
                                          padding_idx=position1_vocab.pad_idx)
        self.position1_emb.weight.data = torch.from_numpy(position1_vocab.embeddings).float()

        self.position2_emb = nn.Embedding(position2_vocab.vocab_size, position2_vocab.emb_size,
                                          padding_idx=position2_vocab.pad_idx)
        self.position2_emb.weight.data = torch.from_numpy(position2_vocab.embeddings).float()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=100,
            kernel_size=(
            3, self.word_emb.embedding_dim + self.position1_emb.embedding_dim + self.position2_emb.embedding_dim),
            stride=1,
            padding=0,
        )

        self.out = nn.Linear(100, relation_vocab.vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, positions1, positions2):  # (bz, seq)
        bz, seq = tokens.size()

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        x = torch.cat((tokens, positions1, positions2), 2).unsqueeze(1)  # (bz, 1, seq, word_emb+pos1_emb+pos2_emb)

        x = F.relu(self.conv1(x).squeeze(-1))
        x = F.max_pool1d(x, seq - 2).squeeze(-1)

        x = self.out(x)
        return x

    def loss(self, gold, pred):
        cost = self.criterion(pred, gold)

        return cost