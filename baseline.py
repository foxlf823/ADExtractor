import torch
import torch.nn as nn
import torch.nn.functional as F
import feature_extractor

class SentimentClassifier(nn.Module):
    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab, et_num_vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'

        self.entity_type_emb = nn.Embedding(entity_type_vocab.vocab_size, entity_type_vocab.emb_size, padding_idx=entity_type_vocab.pad_idx)
        self.entity_type_emb.weight.data = torch.from_numpy(entity_type_vocab.embeddings).float()

        self.entity_emb = nn.Embedding(entity_vocab.vocab_size, entity_vocab.emb_size, padding_idx=entity_vocab.pad_idx)
        self.entity_emb.weight.data = torch.from_numpy(entity_vocab.embeddings).float()

        self.tok_num_betw_emb = nn.Embedding(tok_num_betw_vocab.vocab_size, tok_num_betw_vocab.emb_size, padding_idx=tok_num_betw_vocab.pad_idx)
        self.tok_num_betw_emb.weight.data = torch.from_numpy(tok_num_betw_vocab.embeddings).float()

        self.et_num_emb = nn.Embedding(et_num_vocab.vocab_size, et_num_vocab.emb_size, padding_idx=et_num_vocab.pad_idx)
        self.et_num_emb.weight.data = torch.from_numpy(et_num_vocab.embeddings).float()

        self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

        self.criterion = nn.CrossEntropyLoss()

        self.input_size = context_feature_size+2*entity_type_vocab.emb_size+2*entity_vocab.emb_size + \
                          tok_num_betw_vocab.emb_size + et_num_vocab.emb_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(self.input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, relation_vocab.vocab_size))
        #self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, hidden_features, x2, x1):
        tokens, positions1, positions2, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths = x1

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        return self.net(x)

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)

class MLP(nn.Module):
    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab,  tok_num_betw_vocab,
                                         et_num_vocab,):
        super(MLP, self).__init__()

        self.entity_type_emb = nn.Embedding(entity_type_vocab.vocab_size, entity_type_vocab.emb_size,
                                          padding_idx=entity_type_vocab.pad_idx)
        self.entity_type_emb.weight.data = torch.from_numpy(entity_type_vocab.embeddings).float()

        self.entity_emb = nn.Embedding(entity_vocab.vocab_size, entity_vocab.emb_size,
                                          padding_idx=entity_vocab.pad_idx)
        self.entity_emb.weight.data = torch.from_numpy(entity_vocab.embeddings).float()

        self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

        self.tok_num_betw_emb = nn.Embedding(tok_num_betw_vocab.vocab_size, tok_num_betw_vocab.emb_size, padding_idx=tok_num_betw_vocab.pad_idx)
        self.tok_num_betw_emb.weight.data = torch.from_numpy(tok_num_betw_vocab.embeddings).float()

        self.et_num_emb = nn.Embedding(et_num_vocab.vocab_size, et_num_vocab.emb_size, padding_idx=et_num_vocab.pad_idx)
        self.et_num_emb.weight.data = torch.from_numpy(et_num_vocab.embeddings).float()

        self.input_size = context_feature_size + 2 * entity_type_vocab.emb_size + 2 * entity_vocab.emb_size + \
                          tok_num_betw_vocab.emb_size + et_num_vocab.emb_size

        self.linear = nn.Linear(self.input_size, relation_vocab.vocab_size, bias=False)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, hidden_features, x2, x1):
        _, _, _, _, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths = x1

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        output = self.linear(x)

        return output

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)


