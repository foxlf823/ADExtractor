import torch
import torch.nn.functional as functional
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from options import opt


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 word_vocab, postag_vocab, position1_vocab, position2_vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.word_emb = word_vocab.init_embed_layer()
        self.postag_emb = postag_vocab.init_embed_layer()
        self.position1_emb = position1_vocab.init_embed_layer()
        self.position2_emb = position2_vocab.init_embed_layer()

        self.input_size = word_vocab.emb_size+postag_vocab.emb_size+position1_vocab.emb_size+position2_vocab.emb_size

        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, self.input_size)) for K in kernel_sizes])

        if opt.model_bn:
            self.convs_bn = nn.ModuleList([nn.BatchNorm2d(kernel_num) for K in kernel_sizes])

        # at least 1 hidden layer so that the output size is hidden_size
        assert num_layers > 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                                      nn.Linear(len(kernel_sizes) * kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))

            if opt.model_bn:
                self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, x2, x1):
        tokens, postag, positions1, positions2, e1_token, e2_token = x2

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        postag = self.postag_emb(postag)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        embeds = torch.cat((tokens, postag, positions1, positions2), 2) # (bz, seq, ?)

        # conv
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size
        if opt.model_bn:
            x = [functional.relu(self.convs_bn[i](conv(embeds))).squeeze(3) for i, conv in enumerate(self.convs)]
        else:
            x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 word_vocab, position1_vocab, position2_vocab,
                 num_layers,
                 hidden_size,
                 dropout):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers

        self.hidden_size = hidden_size // 2
        self.n_cells = self.num_layers * 2

        self.word_emb = word_vocab.init_embed_layer()
        self.position1_emb = position1_vocab.init_embed_layer()
        self.position2_emb = position2_vocab.init_embed_layer()

        self.input_size = word_vocab.emb_size+position1_vocab.emb_size+position2_vocab.emb_size

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=True)

        self.attn = DotAttentionLayer(hidden_size)

    def forward(self, x2, x1):
        tokens, positions1, positions2, _, _ = x2
        _, _, _, _, lengths = x1

        lengths_list = lengths.tolist()
        batch_size = tokens.size(0)

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        embeds = torch.cat((tokens, positions1, positions2), 2) # (bz, seq, word_emb+pos1_emb+pos2_emb)

        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = embeds.new(*state_shape)
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
        return self.attn((unpacked_output, lengths))


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = functional.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if torch.cuda.is_available():
            idxes = idxes.cuda(opt.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output