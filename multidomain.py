import logging
from options import opt
from trainandtest import dataset_stat, makeDatasetWithoutUnknown, makeDatasetUnknown
import sortedcontainers
import vocab
import cPickle as pickle
import os
import my_utils
from torch.utils.data import DataLoader
from feature_extractor import *
import capsule
import baseline
import itertools
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

def pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation, test_name,
             other_token, other_entity, other_relation, other_name):
    word_alphabet, postag_alphabet, relation_alphabet, entity_type_alphabet, entity_alphabet = dataset_stat(train_token, train_entity, train_relation)
    logging.info("training dataset stat completed")
    if opt.full_data:
        test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet = dataset_stat(test_token, test_entity, test_relation)
        word_alphabet = word_alphabet | test_word_alphabet
        postag_alphabet = postag_alphabet | test_postag_alphabet
        relation_alphabet = relation_alphabet | test_relation_alphabet
        entity_type_alphabet = entity_type_alphabet | test_entity_type_alphabet
        entity_alphabet = entity_alphabet | test_entity_alphabet
        del test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet
        logging.info("test dataset stat completed")

    for other in other_name:
        other_word_alphabet, other_postag_alphabet, other_relation_alphabet, other_entity_type_alphabet, other_entity_alphabet = dataset_stat(
            other_token[other], other_entity[other], other_relation[other])
        word_alphabet = word_alphabet | other_word_alphabet
        postag_alphabet = postag_alphabet | other_postag_alphabet
        relation_alphabet = relation_alphabet | other_relation_alphabet
        entity_type_alphabet = entity_type_alphabet | other_entity_type_alphabet
        entity_alphabet = entity_alphabet | other_entity_alphabet
        del other_word_alphabet, other_postag_alphabet, other_relation_alphabet, other_entity_type_alphabet, other_entity_alphabet
        logging.info("other {} stat completed".format(other))


    position_alphabet = sortedcontainers.SortedSet()
    for i in range(opt.max_seq_len):
        position_alphabet.add(i)
        position_alphabet.add(-i)

    relation_vocab = vocab.Vocab(relation_alphabet, None, opt.relation_emb_size)
    word_vocab = vocab.Vocab(word_alphabet, opt.emb, opt.word_emb_size)
    postag_vocab = vocab.Vocab(postag_alphabet, None, opt.pos_emb_size)
    entity_type_vocab = vocab.Vocab(entity_type_alphabet, None, opt.entity_type_emb_size)
    entity_vocab = vocab.Vocab(entity_alphabet, None, opt.entity_emb_size)
    position_vocab1 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)
    position_vocab2 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)
    # we directly use position_alphabet to build them, since they are all numbers
    tok_num_betw_vocab = vocab.Vocab(position_alphabet, None, opt.entity_type_emb_size)
    et_num_vocab = vocab.Vocab(position_alphabet, None, opt.entity_type_emb_size)
    logging.info("vocab build completed")

    logging.info("saving ... vocab")
    pickle.dump(word_vocab, open(os.path.join(opt.pretrain, 'word_vocab.pkl'), "wb"), True)
    pickle.dump(postag_vocab, open(os.path.join(opt.pretrain, 'postag_vocab.pkl'), "wb"), True)
    pickle.dump(relation_vocab, open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), "wb"), True)
    pickle.dump(entity_type_vocab, open(os.path.join(opt.pretrain, 'entity_type_vocab.pkl'), "wb"), True)
    pickle.dump(entity_vocab, open(os.path.join(opt.pretrain, 'entity_vocab.pkl'), "wb"), True)
    pickle.dump(position_vocab1, open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), "wb"), True)
    pickle.dump(position_vocab2, open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), "wb"), True)
    pickle.dump(tok_num_betw_vocab, open(os.path.join(opt.pretrain, 'tok_num_betw_vocab.pkl'), "wb"), True)
    pickle.dump(et_num_vocab, open(os.path.join(opt.pretrain, 'et_num_vocab.pkl'), "wb"), True)

    train_X, train_Y, _ = my_utils.getRelationInstance2(train_token, train_entity, train_relation, train_name, word_vocab, postag_vocab,
                                                     relation_vocab, entity_type_vocab,
                                                     entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
    logging.info("training instance build completed, total {}".format(len(train_Y)))
    pickle.dump(train_X, open(os.path.join(opt.pretrain, 'train_X.pkl'), "wb"), True)
    pickle.dump(train_Y, open(os.path.join(opt.pretrain, 'train_Y.pkl'), "wb"), True)


    test_X, test_Y, test_other = my_utils.getRelationInstance2(test_token, test_entity, test_relation, test_name, word_vocab, postag_vocab,
                                                            relation_vocab, entity_type_vocab,
                                                            entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
    logging.info("test instance build completed, total {}".format(len(test_Y)))
    pickle.dump(test_X, open(os.path.join(opt.pretrain, 'test_X.pkl'), "wb"), True)
    pickle.dump(test_Y, open(os.path.join(opt.pretrain, 'test_Y.pkl'), "wb"), True)
    pickle.dump(test_other, open(os.path.join(opt.pretrain, 'test_Other.pkl'), "wb"), True)

    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    postag_vocab = pickle.load(open(os.path.join(opt.pretrain, 'postag_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    entity_type_vocab = pickle.load(open(os.path.join(opt.pretrain, 'entity_type_vocab.pkl'), 'rb'))
    entity_vocab = pickle.load(open(os.path.join(opt.pretrain, 'entity_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))
    tok_num_betw_vocab = pickle.load(open(os.path.join(opt.pretrain, 'tok_num_betw_vocab.pkl'), 'rb'))
    et_num_vocab = pickle.load(open(os.path.join(opt.pretrain, 'et_num_vocab.pkl'), 'rb'))

    for other in other_name:
        other_X, other_Y, _ = my_utils.getRelationInstance2(other_token[other], other_entity[other], other_relation[other], other_name[other],
                                                                       word_vocab, postag_vocab, relation_vocab, entity_type_vocab,
                                                     entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
        logging.info("other {} instance build completed, total {}".format(other, len(other_Y)))
        pickle.dump(other_X, open(os.path.join(opt.pretrain, 'other_{}_X.pkl'.format(other)), "wb"), True)
        pickle.dump(other_Y, open(os.path.join(opt.pretrain, 'other_{}_Y.pkl'.format(other)), "wb"), True)


def train(other_dir):

    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    postag_vocab = pickle.load(open(os.path.join(opt.pretrain, 'postag_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    entity_type_vocab = pickle.load(open(os.path.join(opt.pretrain, 'entity_type_vocab.pkl'), 'rb'))
    entity_vocab = pickle.load(open(os.path.join(opt.pretrain, 'entity_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))
    tok_num_betw_vocab = pickle.load(open(os.path.join(opt.pretrain, 'tok_num_betw_vocab.pkl'), 'rb'))
    et_num_vocab = pickle.load(open(os.path.join(opt.pretrain, 'et_num_vocab.pkl'), 'rb'))


    my_collate = my_utils.sorted_collate if opt.model == 'lstm' else my_utils.unsorted_collate

    # test only on the main domain
    test_X = pickle.load(open(os.path.join(opt.pretrain, 'test_X.pkl'), 'rb'))
    test_Y = pickle.load(open(os.path.join(opt.pretrain, 'test_Y.pkl'), 'rb'))
    test_Other = pickle.load(open(os.path.join(opt.pretrain, 'test_Other.pkl'), 'rb'))
    logging.info("total test instance {}".format(len(test_Y)))
    test_loader = DataLoader(my_utils.RelationDataset(test_X, test_Y),
                              opt.batch_size, shuffle=False, collate_fn=my_collate) # drop_last=True

    # train on the main as well as other domains
    domains = ['main']
    train_loaders, train_iters, unk_loaders, unk_iters = {}, {}, {}, {}
    train_X = pickle.load(open(os.path.join(opt.pretrain, 'train_X.pkl'), 'rb'))
    train_Y = pickle.load(open(os.path.join(opt.pretrain, 'train_Y.pkl'), 'rb'))
    logging.info("total training instance {}".format(len(train_Y)))
    train_loaders['main'], train_iters['main'] = makeDatasetWithoutUnknown(train_X, train_Y, relation_vocab, True, my_collate)
    unk_loaders['main'], unk_iters['main'] = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, opt.unk_ratio)

    for other in other_dir:
        domains.append(other)
        other_X = pickle.load(open(os.path.join(opt.pretrain, 'other_{}_X.pkl'.format(other)), 'rb'))
        other_Y = pickle.load(open(os.path.join(opt.pretrain, 'other_{}_Y.pkl'.format(other)), 'rb'))
        logging.info("other {} instance {}".format(other, len(other_Y)))
        train_loaders[other], train_iters[other] = makeDatasetWithoutUnknown(other_X, other_Y, relation_vocab, True, my_collate)
        unk_loaders[other], unk_iters[other] = makeDatasetUnknown(other_X, other_Y, relation_vocab, my_collate, opt.unk_ratio)

    opt.domains = domains


    F_s = None
    F_d = {}
    if opt.model.lower() == 'lstm':
        F_s = LSTMFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                 opt.F_layers, opt.shared_hidden_size, opt.dropout)
        for domain in opt.domains:
            F_d[domain] = LSTMFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                 opt.F_layers, opt.shared_hidden_size, opt.dropout)
    elif opt.model.lower() == 'cnn':
        F_s = CNNFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
                                                opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
        for domain in opt.domains:
            F_d[domain] = CNNFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
                                                opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise RuntimeError('Unknown feature extractor {}'.format(opt.model))

    if opt.model_high == 'capsule':
        C = capsule.CapsuleNet(2*opt.shared_hidden_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab,
                                         et_num_vocab)
    elif opt.model_high == 'mlp':
        C = baseline.MLP(2*opt.shared_hidden_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab,
                                         et_num_vocab)
    else:
        raise RuntimeError('Unknown model {}'.format(opt.model_high))

    if opt.adv:
        D = DomainClassifier(1, opt.shared_hidden_size, opt.shared_hidden_size,
                             len(opt.domains), opt.loss, opt.dropout, True)

    if torch.cuda.is_available():
        F_s, C = F_s.cuda(opt.gpu), C.cuda(opt.gpu)
        for f_d in F_d.values():
            f_d = f_d.cuda(opt.gpu)
        if opt.adv:
            D = D.cuda(opt.gpu)

    iter_parameter = itertools.chain(
        *map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()]))
    optimizer = optim.Adam(iter_parameter, lr=opt.learning_rate)
    if opt.adv:
        optimizerD = optim.Adam(D.parameters(), lr=opt.learning_rate)


    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(opt.max_epoch):

        F_s.train()
        C.train()
        for f in F_d.values():
            f.train()
        if opt.adv:
            D.train()

        # domain accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        if opt.adv:
            d_correct, d_total = 0, 0

        # conceptually view 1 epoch as 1 epoch of the main domain
        num_iter = len(train_loaders['main'])

        for i in tqdm(range(num_iter)):

            if opt.adv:
                # D iterations
                my_utils.freeze_net(F_s)
                map(my_utils.freeze_net, F_d.values())
                my_utils.freeze_net(C)
                my_utils.unfreeze_net(D)

                if opt.tune_wordemb == False:
                    my_utils.freeze_net(F_s.word_emb)
                    for f_d in F_d.values():
                        my_utils.freeze_net(f_d.word_emb)
                # WGAN n_critic trick since D trains slower
                n_critic = opt.n_critic
                if opt.wgan_trick:
                    if opt.n_critic > 0 and ((epoch == 0 and i < 25) or i % 500 == 0):
                        n_critic = 100

                for _ in range(n_critic):
                    D.zero_grad()

                    # train on both labeled and unlabeled domains
                    for domain in opt.domains:
                        # targets not used
                        x2, x1, _ = my_utils.endless_get_next_batch_without_rebatch(train_loaders[domain],
                                                                                       train_iters[domain])
                        d_targets = my_utils.get_domain_label(opt.loss, domain, len(x2[1]))
                        shared_feat = F_s(x2, x1)
                        d_outputs = D(shared_feat)
                        # D accuracy
                        _, pred = torch.max(d_outputs, 1)
                        d_total += len(x2[1])
                        if opt.loss.lower() == 'l2':
                            _, tgt_indices = torch.max(d_targets, 1)
                            d_correct += (pred == tgt_indices).sum().data.item()
                            l_d = functional.mse_loss(d_outputs, d_targets)
                            l_d.backward()
                        else:
                            d_correct += (pred == d_targets).sum().data.item()
                            l_d = functional.nll_loss(d_outputs, d_targets)
                            l_d.backward()

                    optimizerD.step()

            # F&C iteration
            my_utils.unfreeze_net(F_s)
            map(my_utils.unfreeze_net, F_d.values())
            my_utils.unfreeze_net(C)
            if opt.adv:
                my_utils.freeze_net(D)
            if opt.tune_wordemb == False:
                my_utils.freeze_net(F_s.word_emb)
                for f_d in F_d.values():
                    my_utils.freeze_net(f_d.word_emb)

            F_s.zero_grad()
            for f_d in F_d.values():
                f_d.zero_grad()
            C.zero_grad()

            for domain in opt.domains:

                x2, x1, targets = my_utils.endless_get_next_batch_without_rebatch(train_loaders[domain], train_iters[domain])
                shared_feat = F_s(x2, x1)
                domain_feat = F_d[domain](x2, x1)
                features = torch.cat((shared_feat, domain_feat), dim=1)

                if opt.model_high == 'capsule':
                    c_outputs, x_recon = C.forward(features, x2, x1, targets)
                    l_c = C.loss(targets, c_outputs, features, x2, x1, x_recon, opt.lam_recon)
                elif opt.model_high == 'mlp':
                    c_outputs = C.forward(features, x2, x1)
                    l_c = C.loss(targets, c_outputs)
                else:
                    raise RuntimeError('Unknown model {}'.format(opt.model_high))

                l_c.backward(retain_graph=True)
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().data.item()

                # training with unknown
                x2, x1, targets = my_utils.endless_get_next_batch_without_rebatch(unk_loaders[domain], unk_iters[domain])
                shared_feat = F_s(x2, x1)
                domain_feat = F_d[domain](x2, x1)
                features = torch.cat((shared_feat, domain_feat), dim=1)

                if opt.model_high == 'capsule':
                    c_outputs, x_recon = C.forward(features, x2, x1, targets)
                    l_c = C.loss(targets, c_outputs, features, x2, x1, x_recon, opt.lam_recon)
                elif opt.model_high == 'mlp':
                    c_outputs = C.forward(features, x2, x1)
                    l_c = C.loss(targets, c_outputs)
                else:
                    raise RuntimeError('Unknown model {}'.format(opt.model_high))

                l_c.backward(retain_graph=True)
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().data.item()

            if opt.adv:
                # update F with D gradients on all domains
                for domain in opt.domains:
                    x2, x1, _ = my_utils.endless_get_next_batch_without_rebatch(train_loaders[domain],
                                                                                   train_iters[domain])
                    shared_feat = F_s(x2, x1)
                    d_outputs = D(shared_feat)
                    if opt.loss.lower() == 'gr':
                        d_targets = my_utils.get_domain_label(opt.loss, domain, len(x2[1]))
                        l_d = functional.nll_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= -opt.lambd
                    elif opt.loss.lower() == 'bs':
                        d_targets = my_utils.get_random_domain_label(opt.loss, len(x2[1]))
                        l_d = functional.kl_div(d_outputs, d_targets, size_average=False)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    elif opt.loss.lower() == 'l2':
                        d_targets = my_utils.get_random_domain_label(opt.loss, len(x2[1]))
                        l_d = functional.mse_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    l_d.backward()

            torch.nn.utils.clip_grad_norm_(iter_parameter, opt.grad_clip)
            optimizer.step()

        # regenerate unknown dataset after one epoch
        unk_loaders['main'], unk_iters['main'] = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, opt.unk_ratio)
        for other in other_dir:
            unk_loaders[other], unk_iters[other] = makeDatasetUnknown(other_X, other_Y, relation_vocab, my_collate, opt.unk_ratio)

        logging.info('epoch {} end'.format(epoch))
        if opt.adv and d_total > 0:
            logging.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        logging.info('Training accuracy:')
        logging.info('\t'.join(opt.domains))
        logging.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.domains]))

        test_accuracy = evaluate(F_s, F_d['main'], C, test_loader, test_Other)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(F_s.state_dict(), '{}/F_s.pth'.format(opt.output))
            for d in opt.domains:
                torch.save(F_d[d].state_dict(), '{}/F_d_{}.pth'.format(opt.output, d))
            torch.save(C.state_dict(), '{}/C.pth'.format(opt.output))
            if opt.adv:
                torch.save(D.state_dict(), '{}/D.pth'.format(opt.output))
            pickle.dump(test_Other, open(os.path.join(opt.output, 'results.pkl'), "wb"), True)
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def evaluate(F_s, F_d, C, loader, other):
    F_s.eval()
    F_d.eval()
    C.eval()

    it = iter(loader)
    start, end = 0, 0
    correct = 0
    total = 0
    iii = 0
    for x2, x1, targets in it:

        if targets.size(0) == 1 and (opt.model_bn or opt.model_high_bn):
            for i, _ in enumerate(x2):
                x2[i] = x2[i].expand(2, -1)

            for i, _ in enumerate(x1):
                x1[i] = x1[i].expand(2)


        with torch.no_grad():

            shared_feat = F_s(x2, x1)
            domain_feat = F_d(x2, x1)
            features = torch.cat((shared_feat, domain_feat), dim=1)

            if opt.model_high == 'capsule':
                c_outputs, _ = C.forward(features, x2, x1)
            elif opt.model_high == 'mlp':
                c_outputs = C.forward(features, x2, x1)

            else:
                raise RuntimeError('Unknown model {}'.format(opt.model_high))

            _, pred = torch.max(c_outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

        start = end
        end = end + targets.size(0)

        for i, d in enumerate(other[start:end]):
            d["type"] = pred[i].item()

        iii += 1

    acc = 100.0 * correct / total
    return acc


class DomainClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_domains,
                 loss_type,
                 dropout,
                 batch_norm=False):
        super(DomainClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_domains = num_domains
        self.loss_type = loss_type
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        if loss_type.lower() == 'gr' or loss_type.lower() == 'bs':
            self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        scores = self.net(input)
        if self.loss_type.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores