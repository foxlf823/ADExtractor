import torch
import torch.nn as nn
import torch.nn.functional as F
from options import opt


class CapsuleNet_EM(nn.Module):
    def __init__(self, input_size, relation_vocab):
        super(CapsuleNet_EM, self).__init__()

        self.primary_caps = PrimaryCaps(16, 16)
        self.class_capsule = ClassCaps(16, relation_vocab.vocab_size, 3)

    def forward(self, x):
        x = x.view(-1, 16, 4, 4)

        pose, activation = self.primary_caps(x)  # (bs, 16, 4, 4, 4, 4) (bs, 16, 4, 4)
        pose, activation = self.class_capsule(pose, activation)  # (bs, 10, 4, 4) (bs, 10)
        return activation

    def loss(self, by, y_pred):

        return self._spread_loss(y_pred, by)


    def _spread_loss(self, activation, target):
        '''
        activation: (bs, class_num)
        target: class_id LongTensor such as [1, 4, 9, ...]
        '''
        bs = activation.size(0)
        class_num = activation.size(1)

        y_gold = torch.FloatTensor(bs, class_num)
        if torch.cuda.is_available():
            y_gold = y_gold.cuda(opt.gpu)

        y_gold = y_gold.zero_().scatter_(1, target.unsqueeze(1), 9999)
        loss = torch.clamp(0.2 - (y_gold - activation), 0, 9999)
        loss = torch.sum(torch.pow(loss, 2), dim=1, keepdim=False).mean()

        return loss


class PrimaryCaps(nn.Module):
    def __init__(self, A, B):
        super(PrimaryCaps, self).__init__()

        self.pose_size = torch.Size([4, 4])
        self.B = B

        self.conv_pose = nn.Conv2d(A, B * self.pose_size[0] * self.pose_size[1], 1, 1, 0)
        self.conv_activation = nn.Conv2d(A, B, 1, 1, 0)
        self.conv_activation_bn = nn.BatchNorm2d(B)

    def forward(self, x):  # (bs, 32, 14, 14)
        bs = x.size(0)

        pose = self.conv_pose(x)  # (bs, 16*4*4=256, 14, 14)
        pose = pose.view(bs, self.B, self.pose_size[0], self.pose_size[1], pose.size(-2), pose.size(-1))

        activation = F.sigmoid(self.conv_activation_bn(self.conv_activation(x)))  # (bs, 16, 14, 14)

        return pose, activation


class ClassCaps(nn.Module):
    def __init__(self, in_cap_num, out_cap_num, iterations):
        super(ClassCaps, self).__init__()

        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.iterations = iterations
        self.kernel_size = 1
        self.pose_size = 4  # 4x4

        self.kernel_cap_num = self.in_cap_num * self.kernel_size * self.kernel_size

        self.W = nn.Parameter(torch.randn(self.kernel_cap_num, self.out_cap_num, self.pose_size, self.pose_size))

        self.beta_u = nn.Parameter(torch.randn(self.out_cap_num))
        self.beta_a = nn.Parameter(torch.randn(self.out_cap_num))

        self.it_min = 1.0
        self.it_max = min(iterations, 3.0)
        self.epsilon = 1e-9

    def forward(self, pose, activation):  # (bs, 16, 4, 4, 4, 4) (bs, 16, 4, 4)

        bs = pose.size(0)

        pose = pose.permute(0, 4, 5, 1, 2, 3)  # (bs, 4, 4, 16, 4, 4)
        pose = pose.unsqueeze(dim=4)  # (bs, 4, 4, 16, 1, 4, 4)
        W = self.W.view(1, 1, 1, self.kernel_cap_num, self.out_cap_num, self.pose_size, self.pose_size)
        vote = torch.matmul(pose, W)  # (bs, 4, 4, 16, 10, 4, 4)
        vote = vote.view(bs, -1, vote.size(4), self.pose_size * self.pose_size)  # (bs, 256, 10, 16)

        activation = activation.contiguous().view(bs, -1)  # (bs, 256)

        # poses: (bs, 10, 16) , activations: (bs, 10)
        pose, activation = self.matrix_capsules_em_routing2(bs,
            vote, activation, self.beta_u, self.beta_a, self.iterations)

        pose = pose.view(bs, self.out_cap_num, self.pose_size, self.pose_size)
        # poses: (bs, 10, 4, 4) , activations: (bs, 10)
        return pose, activation


    def matrix_capsules_em_routing2(self, bs, votes, i_activations, beta_v, beta_a, iterations):
        '''
        For ConvCaps, the dimensions of parameters are former.
        For ClassCaps, the dimensions of parameters are latter.
        votes: (bs, spatial, spatial, incap, outcap, cap_dim) or (bs, incap, outcap, capdim)
        i_activations: (bs, spatial, spatial, incap) or  (bs, incap)
        beta_v, beta_a: (outcap)

        return:
        poses: (bs, spatial, spatial, outcap, cap_dim) or (bs, outcap, cap_dim)
        activations: (bs, spatial, spatial, outcap) or (bs, outcap)
        '''

        kernel_and_incap = votes.size(1)
        outcap = votes.size(2)
        rr = torch.ones(bs, kernel_and_incap, outcap)
        if torch.cuda.is_available():
            rr = rr.cuda(opt.gpu)

        rr = rr / outcap
        beta_v = beta_v.view(1, 1, outcap, 1)
        beta_a = beta_a.view(1, outcap)



        for it in range(iterations):

            inverse_temperature = self.it_min + (self.it_max - self.it_min) * it / max(1.0, iterations - 1.0)
            # m-step
            rr_prime = (rr * i_activations.unsqueeze(-1))  # (bs, spatial, spatial, incap, outcap)
            rr_prime_sum = rr_prime.sum(-2, keepdim=True).unsqueeze(-1)  # (bs, spatial, spatial, 1, outcap, 1)
            o_mean = ((rr_prime.unsqueeze(-1) * votes).sum(-3, keepdim=True) / (
                        rr_prime_sum + self.epsilon))  # (bs, spatial, spatial, 1, outcap, capdim)
            o_stdv = ((rr_prime.unsqueeze(-1) * (votes - o_mean) ** 2).sum(-3, keepdim=True) / (
                        rr_prime_sum + self.epsilon)) ** (1 / 2)  # (bs, spatial, spatial, 1, outcap, capdim)
            o_cost = ((beta_v + torch.log(o_stdv + self.epsilon)) * rr_prime_sum).sum(
                -1).squeeze()  # (bs, spatial, spatial, outcap)
            # For numeric stability.
            o_cost_mean = torch.mean(o_cost, dim=-1, keepdim=True)
            o_cost_stdv = torch.sqrt(torch.sum((o_cost - o_cost_mean) ** 2, dim=-1, keepdim=True) / outcap)
            # (bs, spatial, spatial, outcap)
            o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + self.epsilon)
            o_activations = torch.sigmoid(inverse_temperature * o_activations_cost)

            # e-step
            if it < iterations - 1:
                mu, sigma_square, V_, a__ = o_mean.data, o_stdv.data, votes.data, o_activations.data
                normal = torch.distributions.Normal(mu, sigma_square)
                p = torch.exp(normal.log_prob(V_))  # (bs, spatial, spatial, incap, outcap, capdim)
                ap = a__.unsqueeze(-2) * p.sum(-1, keepdim=False)  # (bs, spatial, spatial, incap, outcap)
                rr = (ap / (ap.sum(-1, keepdim=True) + self.epsilon)).clone()

        return o_mean.squeeze(), o_activations






