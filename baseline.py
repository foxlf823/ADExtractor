import torch
import torch.nn as nn
import torch.nn.functional as F
import feature_extractor

class MLP(nn.Module):
    def __init__(self, input_size, relation_vocab, entity_type_vocab, entity_vocab):
        super(MLP, self).__init__()

        self.linear = nn.Linear(input_size+2*entity_type_vocab.emb_size+2*entity_vocab.emb_size, relation_vocab.vocab_size, bias=False)

        self.entity_type_emb = nn.Embedding(entity_type_vocab.vocab_size, entity_type_vocab.emb_size,
                                          padding_idx=entity_type_vocab.pad_idx)
        self.entity_type_emb.weight.data = torch.from_numpy(entity_type_vocab.embeddings).float()

        self.entity_emb = nn.Embedding(entity_vocab.vocab_size, entity_vocab.emb_size,
                                          padding_idx=entity_vocab.pad_idx)
        self.entity_emb.weight.data = torch.from_numpy(entity_vocab.embeddings).float()

        self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, hidden_features, x2, x1):
        tokens, positions1, positions2, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, lengths = x1

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2), dim=1)
        output = self.linear(x)

        return output

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)


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