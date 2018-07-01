import torch
import torch.nn as nn
import torch.nn.functional as F
import feature_extractor
from options import opt


class MLP(nn.Module):
    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab,  tok_num_betw_vocab,
                                         et_num_vocab):
        super(MLP, self).__init__()

        if not opt.onlyuse_seqfeature:
            self.entity_type_emb = entity_type_vocab.init_embed_layer()

            self.entity_emb = entity_vocab.init_embed_layer()

            self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

            self.tok_num_betw_emb = tok_num_betw_vocab.init_embed_layer()

            self.et_num_emb = et_num_vocab.init_embed_layer()

            self.input_size = context_feature_size + 2 * entity_type_vocab.emb_size + 2 * entity_vocab.emb_size + \
                              tok_num_betw_vocab.emb_size + et_num_vocab.emb_size
        else:
            self.input_size = context_feature_size

        self.linear = nn.Linear(self.input_size, relation_vocab.vocab_size, bias=False)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, hidden_features, x2, x1):
        _, _, _, _, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths = x1

        if not opt.onlyuse_seqfeature:
            e1_t = self.entity_type_emb(e1_type)
            e2_t = self.entity_type_emb(e2_type)

            e1 = self.entity_emb(e1_token)
            e1 = self.dot_att((e1, e1_length))
            e2 = self.entity_emb(e2_token)
            e2 = self.dot_att((e2, e2_length))

            v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

            v_et_num = self.et_num_emb(et_num)

            x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)
        else:
            x = hidden_features

        output = self.linear(x)

        return output

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)


