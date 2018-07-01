import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt
import feature_extractor

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

        if opt.model_high_bn:
            self.batchnorm = nn.BatchNorm2d(out_num_caps)

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
            b = b.cuda(opt.gpu)

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                if opt.model_high_bn:
                    outputs = squash(self.batchnorm(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True)))
                else:
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                if opt.model_high_bn:
                    outputs = squash(self.batchnorm(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True)))
                else:
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


# class PrimaryCapsule(nn.Module):
#
#     def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
#         super(PrimaryCapsule, self).__init__()
#         self.dim_caps = dim_caps
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         if opt.model_high_bn:
#             self.conv2d_bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         outputs = self.conv2d(x)
#         if opt.model_high_bn:
#             outputs = self.conv2d_bn(outputs)
#         outputs = outputs.view(x.size(0), -1, self.dim_caps)
#         return squash(outputs)

class PrimaryCapsule(nn.Module):

    def __init__(self, dim_caps, stride):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv = nn.Conv1d(self.dim_caps, self.dim_caps, kernel_size=1, stride=stride, padding=0)
        if opt.model_high_bn:
            self.conv_bn = nn.BatchNorm1d(self.dim_caps)

    def forward(self, x):
        x = x.view(x.size(0), self.dim_caps, -1)
        outputs = self.conv(x)
        if opt.model_high_bn:
            outputs = self.conv_bn(outputs)
        outputs = outputs.permute(0, 2, 1)
        return squash(outputs)

class CapsuleNet(nn.Module):

    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab,
                                         et_num_vocab):
        super(CapsuleNet, self).__init__()

        self.input_cap_dim = opt.init_dim_cap
        assert entity_type_vocab.emb_size == entity_vocab.emb_size
        assert entity_type_vocab.emb_size == tok_num_betw_vocab.emb_size
        assert entity_type_vocab.emb_size == et_num_vocab.emb_size

        # context capsule setting
        self.context_cap_stride = 1
        self.context_caps = PrimaryCapsule(self.input_cap_dim, self.context_cap_stride)
        assert context_feature_size % (self.input_cap_dim * self.context_cap_stride) == 0
        self.context_cap_num = context_feature_size // (self.input_cap_dim * self.context_cap_stride)

        # other features just reshape
        assert entity_type_vocab.emb_size % self.input_cap_dim == 0

        if not opt.onlyuse_seqfeature:
            self.entity_type_emb = entity_type_vocab.init_embed_layer()

            self.entity_emb = entity_vocab.init_embed_layer()
            self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

            self.tok_num_betw_emb = tok_num_betw_vocab.init_embed_layer()

            self.et_num_emb = et_num_vocab.init_embed_layer()

            # dense capsule settings
            # context capsule number + other capsule number
            self.input_cap_num = self.context_cap_num + 6 * (entity_type_vocab.emb_size // self.input_cap_dim)
        else:
            self.input_cap_num = self.context_cap_num


        self.digitcaps = DenseCapsule(in_num_caps=self.input_cap_num, in_dim_caps=self.input_cap_dim,
                                      out_num_caps=relation_vocab.vocab_size, out_dim_caps=16, routings=3)


        # output
        self.class_num = relation_vocab.vocab_size

        if opt.reconstruct:
            if not opt.onlyuse_seqfeature:
                self.input_size = context_feature_size + 6*entity_type_vocab.emb_size
            else:
                self.input_size = context_feature_size
            self.decoder = nn.Sequential(
                nn.Linear(16 * relation_vocab.vocab_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.input_size),
                nn.Sigmoid()
            )
            self.recon_loss = nn.MSELoss()

    def forward(self, hidden_features, x2, x1, y=None):

        _, _, _, _, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths = x1

        bz = hidden_features.size(0)

        if not opt.onlyuse_seqfeature:
            e1_t = self.entity_type_emb(e1_type)
            e2_t = self.entity_type_emb(e2_type)

            e1 = self.entity_emb(e1_token)
            e1 = self.dot_att((e1, e1_length))
            e2 = self.entity_emb(e2_token)
            e2 = self.dot_att((e2, e2_length))

            v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)
            v_et_num = self.et_num_emb(et_num)


        # 1
        # hidden_features = hidden_features.view(bz, -1, self.primary_cap_dim)
        # e1_t = e1_t.view(bz, -1, self.primary_cap_dim)
        # e2_t = e2_t.view(bz, -1, self.primary_cap_dim)
        # e1 = e1.view(bz, -1, self.primary_cap_dim)
        # e2 = e2.view(bz, -1, self.primary_cap_dim)
        # x = torch.cat((hidden_features, e1_t, e2_t, e1, e2), dim=1)
        # x = squash(x)

        # 2
        # hidden_features = hidden_features.view(-1, 16, 4, 4)
        # hidden_features = self.primarycaps(hidden_features)
        # e1_t = e1_t.view(bz, -1, self.primary_cap_dim)
        # e2_t = e2_t.view(bz, -1, self.primary_cap_dim)
        # e1 = e1.view(bz, -1, self.primary_cap_dim)
        # e2 = e2.view(bz, -1, self.primary_cap_dim)
        # x = torch.cat((e1_t, e2_t, e1, e2), dim=1)
        # x = squash(x)
        # x = torch.cat((hidden_features, x), dim=1)

        # 3
        # x = torch.cat((hidden_features, e1_t, e2_t, e1, e2), dim=1)
        # x = x.view(bz, 16, 3, 6)
        # x = self.primarycaps(x)


        hidden_features = self.context_caps(hidden_features)

        if not opt.onlyuse_seqfeature:
            e1_t = e1_t.view(bz, -1, self.input_cap_dim)
            e2_t = e2_t.view(bz, -1, self.input_cap_dim)
            e1 = e1.view(bz, -1, self.input_cap_dim)
            e2 = e2.view(bz, -1, self.input_cap_dim)
            v_tok_num_betw = v_tok_num_betw.view(bz, -1, self.input_cap_dim)
            v_et_num = v_et_num.view(bz, -1, self.input_cap_dim)
            x = torch.cat((e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)
            x = squash(x)

            x = torch.cat((hidden_features, x), dim=1)
        else:
            x = hidden_features


        x = self.digitcaps(x)
        length = x.norm(dim=-1)

        if opt.reconstruct and (y is not None):
            y = one_hot1(y.view(bz, 1), self.class_num)
            reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
            return length, reconstruction
        else:
            return length, None



    def loss(self, by, y_pred, hidden_features, x2, x1, x_recon, lam_recon):

       return self._caps_loss(by, y_pred, hidden_features, x2, x1, x_recon, lam_recon)


    def _caps_loss(self, by, y_pred, hidden_features, x2, x1, x_recon, lam_recon):

        y_true = torch.zeros(y_pred.size())
        if torch.cuda.is_available():
            y_true = y_true.cuda(opt.gpu)
        y_true.scatter_(1, by.view(-1, 1), 1.)

        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        if opt.reconstruct:

            with torch.no_grad():
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
                    x = hidden_features.clone()

            L_recon = self.recon_loss(x_recon, x)

            return L_margin + lam_recon * L_recon
        else:
            return L_margin




def one_hot1(indices, depth, value=1):

    onehot = torch.FloatTensor(indices.size(0), depth)
    if torch.cuda.is_available():
        onehot = onehot.cuda(opt.gpu)
    onehot.zero_()
    onehot.scatter_(1, indices, value)
    return onehot

