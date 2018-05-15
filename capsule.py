import torch
import torch.nn as nn
import torch.nn.functional as F



def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    #         stdv = 1. / math.sqrt(in_dim_caps*in_num_caps)
    #         self.weight = nn.Parameter(torch.zeros(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
    #         self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        #         b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()
        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)
        if torch.cuda.is_available():
            b = b.cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)



class CapsuleNet(nn.Module):

    def __init__(self, input_size, dim_enlarge_rate, init_dim_cap, relation_vocab):
        super(CapsuleNet, self).__init__()

        assert input_size%init_dim_cap==0, "input_size should be divided by init_dim_cap"
        self.dim_enlarge_rate = dim_enlarge_rate
        self.init_dim_cap = init_dim_cap
        self.init_num_cap = input_size // init_dim_cap

        in_dim_caps = self.init_dim_cap
        cnt = 0
        while input_size//(self.dim_enlarge_rate*in_dim_caps) > relation_vocab.vocab_size:
            in_dim_caps *= self.dim_enlarge_rate
            cnt += 1
        assert cnt > 0, "should have at least one capsule layer"

        in_dim_caps = self.init_dim_cap
        self.caplayers = nn.ModuleList()
        #self.caplayers = nn.Sequential()
        for i in range(cnt):
            if i == cnt-1:
                self.caplayers.add_module('f-capsule-{}'.format(i),
                                          DenseCapsule(in_num_caps=input_size // in_dim_caps,
                                                       in_dim_caps=in_dim_caps,
                                                       out_num_caps=relation_vocab.vocab_size,
                                                       out_dim_caps=self.dim_enlarge_rate * in_dim_caps,
                                                       routings=3))
            else:
                self.caplayers.add_module('f-capsule-{}'.format(i),
                                          DenseCapsule(in_num_caps=input_size // in_dim_caps,
                                                       in_dim_caps=in_dim_caps,
                                                       out_num_caps=input_size // (self.dim_enlarge_rate * in_dim_caps),
                                                       out_dim_caps=self.dim_enlarge_rate * in_dim_caps,
                                                       routings=3))
            in_dim_caps *= self.dim_enlarge_rate

        # self.test = nn.Linear(128, 9)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):  # (bz, input_size)
        x = x.view(-1, self.init_num_cap, self.init_dim_cap)
        #x = squash(x)


        #x = self.caplayers(x)  # [bz, classes, cap_dim]
        for cap in self.caplayers:
            x = cap(x)

        x = x.norm(dim=-1)

        # x = self.test(x)

        return x

    def loss(self, by, y_pred):

       return self._caps_loss(by, y_pred)
        # return  self.criterion(y_pred, by)


    def _caps_loss(self, by, y_pred):

        y_true = torch.zeros(y_pred.size())
        if torch.cuda.is_available():
            y_true = y_true.cuda()
        y_true.scatter_(1, by.view(-1, 1), 1.)

        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        return L_margin


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param num_caps: number of capsule
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, num_caps, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.num_caps = num_caps
        self.conv2d = nn.Conv2d(in_channels, dim_caps*num_caps, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x): # (bz, in_channels, ?, ?)
        outputs = self.conv2d(x) # (bz, dim_caps*num_cap, ?, ?)
        outputs = outputs.view(x.size(0), self.num_caps*self.dim_caps, -1)
        outputs = F.max_pool1d(outputs, outputs.size(-1)).squeeze(-1) # eliminate the last dim, in order to keep self.num_caps*self.dim_caps
        outputs = outputs.view(x.size(0), self.num_caps, self.dim_caps)
        return squash(outputs)


class CapsuleNet1(nn.Module):

    def __init__(self, word_vocab, position1_vocab, position2_vocab, relation_vocab):
        super(CapsuleNet1, self).__init__()

        self.word_emb = nn.Embedding(word_vocab.vocab_size, word_vocab.emb_size, padding_idx=word_vocab.pad_idx)
        self.word_emb.weight.data = torch.from_numpy(word_vocab.embeddings).float()

        self.position1_emb = nn.Embedding(position1_vocab.vocab_size, position1_vocab.emb_size,
                                          padding_idx=position1_vocab.pad_idx)
        self.position1_emb.weight.data = torch.from_numpy(position1_vocab.embeddings).float()

        self.position2_emb = nn.Embedding(position2_vocab.vocab_size, position2_vocab.emb_size,
                                          padding_idx=position2_vocab.pad_idx)
        self.position2_emb.weight.data = torch.from_numpy(position2_vocab.embeddings).float()

        # Layer 1: Just a conventional Conv2D layer
        self.conv1_out_channel = 256  # default 256
        self.conv1 = nn.Conv2d(1, self.conv1_out_channel,
                               kernel_size=(3, self.word_emb.embedding_dim + self.position1_emb.embedding_dim + self.position2_emb.embedding_dim),
                               stride=1, padding=0)


        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        # Since the sequence length varies in each batch, the left sequence length varies after convolution.
        # Therefore, we need to transform such variable-length sequence into fixed-length.
        # Common method will be pooling or attention, here we choose pooling for simplicity. See PrimaryCapsule implement.
        self.primarycap_dim = 8  # default 8
        self.primarycap_num = 32 # default 32
        self.primarycaps = PrimaryCapsule(self.conv1_out_channel, self.primarycap_num, self.primarycap_dim,
                                          kernel_size=(3, 1), stride=2, padding=0)

        # Layer 4: Capsule layer. Routing algorithm works here.
        self.densecap_dim = 16  # default 16
        self.digitcaps = DenseCapsule(in_num_caps=self.primarycap_num,
                                      in_dim_caps=self.primarycap_dim,
                                      out_num_caps=relation_vocab.vocab_size, out_dim_caps=self.densecap_dim, routings=3)


    def forward(self, tokens, positions1, positions2):  # (bz, seq)

        bz, seq = tokens.size()

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        x = torch.cat((tokens, positions1, positions2), 2).unsqueeze(1)  # (bz, 1, seq, word_emb+pos1_emb+pos2_emb)

        x = F.relu(self.conv1(x)) # (bz, self.conv1_out_channel, ? , 1)

        x = self.primarycaps(x) # (bz, self.primarycap_num, self.primarycap_dim)

        x = self.digitcaps(x)  # [bz, classes, self.densecap_dim]

        x = x.norm(dim=-1)

        return x

    def loss(self, by, y_pred):

        return self._caps_loss(by, y_pred)


    def _caps_loss(self, by, y_pred):

        y_true = torch.zeros(y_pred.size())
        if torch.cuda.is_available():
            y_true = y_true.cuda()
        y_true.scatter_(1, by.view(-1, 1), 1.)

        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        return L_margin
