import utils
import vocab
import torch
from torch.utils.data import DataLoader
from options import opt
from tqdm import tqdm
import logging
import baseline
import torch.optim as optim
import os
import cPickle as pickle
import sortedcontainers
import capsule
import bioc
from feature_extractor import *
import itertools
import math
import capsule_em

def dataset_stat(tokens, entities, relations):
    word_alphabet = sortedcontainers.SortedSet()
    # relation may inter-sentence, so stat position based on sentence firstly, and relation instance secondly.
    position_alphabet = sortedcontainers.SortedSet()
    max_sequence_length = 0

    relation_alphabet = sortedcontainers.SortedSet()

    for i, doc_token in enumerate(tokens):

        sent_idx = 0
        sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]
        while sentence.shape[0] != 0:
            for _, token in sentence.iterrows():
                word_alphabet.add(utils.normalizeWord(token['text']))
            if sentence.shape[0] > max_sequence_length:
                max_sequence_length = sentence.shape[0]
            sent_idx += 1
            sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]


    for i, doc_relation in enumerate(relations):

        doc_token = tokens[i]
        doc_entity = entities[i]

        for index, relation in doc_relation.iterrows():

            relation_alphabet.add(relation['type'])

            # find entity mention
            entity1 = doc_entity[(doc_entity['id']==relation['entity1_id'])].iloc[0]
            entity2 = doc_entity[(doc_entity['id'] == relation['entity2_id'])].iloc[0]
            # find all sentences between entity1 and entity2
            former = entity1 if entity1['start']<entity2['start'] else entity2
            latter = entity2 if entity1['start']<entity2['start'] else entity1
            context_token = doc_token[(doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
            if context_token.shape[0] > max_sequence_length:
                max_sequence_length = context_token.shape[0]

    for i in range(max_sequence_length):
        position_alphabet.add(i)
        position_alphabet.add(-i)

    return word_alphabet, position_alphabet, relation_alphabet




def test(test_token, test_entity, test_relation, test_name, result_dumpdir):
    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    logging.info("loading ... model")
    if opt.model.lower() == 'lstm':
        feature_extractor = LSTMFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                 opt.F_layers, opt.shared_hidden_size, opt.dropout)
    elif opt.model.lower() == 'cnn':
        feature_extractor = CNNFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise RuntimeError('Unknown feature extractor {}'.format(opt.model))
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda(opt.gpu)

    if opt.model_high == 'capsule':
        # m = capsule.CapsuleNet(opt.shared_hidden_size, opt.dim_enlarge_rate, opt.init_dim_cap, relation_vocab)
        m = capsule.CapsuleNet(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'capsule_em':
        m = capsule_em.CapsuleNet_EM(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'mlp':
        m = baseline.MLP(opt.shared_hidden_size, relation_vocab)
    else:
        raise RuntimeError('Unknown model {}'.format(opt.model_high))
    if torch.cuda.is_available():
        m = m.cuda(opt.gpu)

    feature_extractor.load_state_dict(torch.load(os.path.join(opt.output, 'feature_extractor.pth')))
    m.load_state_dict(torch.load(os.path.join(opt.output, 'model.pth')))
    m.eval()

    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate

    with torch.no_grad():

        for i in tqdm(range(len(test_relation))): # this procedure should keep consistent with utils.getRelatonInstance

            doc_relation = test_relation[i]
            doc_token = test_token[i]
            doc_entity = test_entity[i]
            doc_name = test_name[i]

            collection = bioc.BioCCollection()
            document = bioc.BioCDocument()
            collection.add_document(document)
            document.id = doc_name
            passage = bioc.BioCPassage()
            document.add_passage(passage)
            passage.offset = 0
            
            for _, entity in doc_entity.iterrows():
                anno_entity = bioc.BioCAnnotation()
                passage.add_annotation(anno_entity)
                anno_entity.id = entity['id']
                anno_entity.infons['type'] = entity['type']
                anno_entity_location = bioc.BioCLocation(entity['start'], entity['end']-entity['start'])
                anno_entity.add_location(anno_entity_location)
                anno_entity.text = entity['text']

            for _, relation in doc_relation.iterrows():

                # find entity mention
                entity1 = doc_entity[(doc_entity['id']==relation['entity1_id'])].iloc[0]
                entity2 = doc_entity[(doc_entity['id'] == relation['entity2_id'])].iloc[0]
                # find all sentences between entity1 and entity2
                former = entity1 if entity1['start']<entity2['start'] else entity2
                latter = entity2 if entity1['start']<entity2['start'] else entity1
                context_token = doc_token[(doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
                words = []
                positions1 = []
                positions2 = []
                i = 0
                former_head = -1
                latter_head = -1
                for _, token in context_token.iterrows():
                    if token['start'] >= former['start'] and token['end'] <= former['end']:
                        if former_head < i:
                            former_head = i
                    if token['start'] >= latter['start'] and token['end'] <= latter['end']:
                        if latter_head < i:
                            latter_head = i

                    i += 1

                if former_head == -1: # due to tokenization error, e.g., 10_197, hyper-CVAD-based vs hyper-CVAD
                    logging.debug('former_head not found, entity {} {} {}'.format(former['id'], former['start'], former['text']))
                    continue
                if latter_head == -1:
                    logging.debug('latter_head not found, entity {} {} {}'.format(latter['id'], latter['start'], latter['text']))
                    continue


                i = 0
                for _, token in context_token.iterrows():

                    word = utils.normalizeWord(token['text'])
                    words.append(word_vocab.lookup(word))

                    positions1.append(position_vocab1.lookup(former_head-i))
                    positions2.append(position_vocab2.lookup(latter_head-i))

                    i += 1


                # here we ignore utils.RelationDataset(test_X, test_Y, opt.max_seq_len)
                # [({'tokens': [171, 35, 371, 304, 6, 243, 389, 106, 2],
                #    'positions2': [107, 106, 105, 104, 103, 102, 101, 100, 99],
                #    'positions1': [105, 104, 103, 102, 101, 100, 99, 98, 97]}, 3), []]

                batch = [({'tokens': words, 'positions1': positions1, 'positions2': positions2},-1)]
                tokens, positions1, positions2, lengths, y = my_collate(batch)

                hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
                outputs = m.forward(hidden_features)
                _, pred = torch.max(outputs, 1)

                # anno_former = bioc.BioCAnnotation()
                # passage.add_annotation(anno_former)
                # anno_former.id = former['id']
                # anno_former.infons['type'] = former['type']
                # anno_former_location = bioc.BioCLocation(former['start'], former['end']-former['start'])
                # anno_former.add_location(anno_former_location)
                # anno_former.text = former['text']
                #
                # anno_latter = bioc.BioCAnnotation()
                # passage.add_annotation(anno_latter)
                # anno_latter.id = latter['id']
                # anno_latter.infons['type'] = latter['type']
                # anno_latter_location = bioc.BioCLocation(latter['start'], latter['end']-latter['start'])
                # anno_latter.add_location(anno_latter_location)
                # anno_latter.text = latter['text']

                bioc_relation = bioc.BioCRelation()
                passage.add_relation(bioc_relation)
                bioc_relation.id = relation['id']
                bioc_relation.infons['type'] = relation_vocab.lookup_id2str(pred.item())

                node1 = bioc.BioCNode(former['id'], 'annotation 1')
                bioc_relation.add_node(node1)
                node2 = bioc.BioCNode(latter['id'], 'annotation 2')
                bioc_relation.add_node(node2)


            with open(os.path.join(result_dumpdir, doc_name+".bioc.xml"), 'w') as fp:
                bioc.dump(collection, fp)


def test1(test_token, test_entity, test_relation, test_name, result_dumpdir):
    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    logging.info("loading ... model")
    if opt.model.lower() == 'lstm':
        feature_extractor = LSTMFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                 opt.F_layers, opt.shared_hidden_size, opt.dropout)
    elif opt.model.lower() == 'cnn':
        feature_extractor = CNNFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise RuntimeError('Unknown feature extractor {}'.format(opt.model))
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda(opt.gpu)

    if opt.model_high == 'capsule':
        # m = capsule.CapsuleNet(opt.shared_hidden_size, opt.dim_enlarge_rate, opt.init_dim_cap, relation_vocab)
        m = capsule.CapsuleNet(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'capsule_em':
        m = capsule_em.CapsuleNet_EM(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'mlp':
        m = baseline.MLP(opt.shared_hidden_size, relation_vocab)
    else:
        raise RuntimeError('Unknown model {}'.format(opt.model_high))
    if torch.cuda.is_available():
        m = m.cuda(opt.gpu)

    feature_extractor.load_state_dict(torch.load(os.path.join(opt.output, 'feature_extractor.pth')))
    m.load_state_dict(torch.load(os.path.join(opt.output, 'model.pth')))
    m.eval()

    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate

    for i in tqdm(range(len(test_relation))):  # this procedure should keep consistent with utils.getRelationInstance1

        doc_relation = test_relation[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]
        doc_name = test_name[i]

        collection = bioc.BioCCollection()
        document = bioc.BioCDocument()
        collection.add_document(document)
        document.id = doc_name
        passage = bioc.BioCPassage()
        document.add_passage(passage)
        passage.offset = 0

        for _, entity in doc_entity.iterrows():
            anno_entity = bioc.BioCAnnotation()
            passage.add_annotation(anno_entity)
            anno_entity.id = entity['id']
            anno_entity.infons['type'] = entity['type']
            anno_entity_location = bioc.BioCLocation(entity['start'], entity['end'] - entity['start'])
            anno_entity.add_location(anno_entity_location)
            anno_entity.text = entity['text']

        row_num = doc_entity.shape[0]
        relation_id = 1
        for latter_idx in range(row_num):

            for former_idx in range(row_num):

                if former_idx < latter_idx:

                    former = doc_entity.loc[former_idx]
                    latter = doc_entity.loc[latter_idx]

                    if math.fabs(latter['sent_idx']-former['sent_idx']) >= opt.sent_window:
                        continue

                    if utils.relationConstraint(former, latter) == False:
                        continue

                    context_token = doc_token[(doc_token['sent_idx'] >= former['sent_idx']) & (
                                doc_token['sent_idx'] <= latter['sent_idx'])]
                    words = []
                    positions1 = []
                    positions2 = []
                    i = 0
                    former_head = -1
                    latter_head = -1
                    for _, token in context_token.iterrows():
                        if token['start'] >= former['start'] and token['end'] <= former['end']:
                            if former_head < i:
                                former_head = i
                        if token['start'] >= latter['start'] and token['end'] <= latter['end']:
                            if latter_head < i:
                                latter_head = i

                        i += 1

                    if former_head == -1:  # due to tokenization error, e.g., 10_197, hyper-CVAD-based vs hyper-CVAD
                        logging.debug('former_head not found, entity {} {} {}'.format(former['id'], former['start'],
                                                                                      former['text']))
                        continue
                    if latter_head == -1:
                        logging.debug('latter_head not found, entity {} {} {}'.format(latter['id'], latter['start'],
                                                                                      latter['text']))
                        continue

                    i = 0
                    for _, token in context_token.iterrows():
                        word = utils.normalizeWord(token['text'])
                        words.append(word_vocab.lookup(word))

                        positions1.append(position_vocab1.lookup(former_head - i))
                        positions2.append(position_vocab2.lookup(latter_head - i))

                        i += 1


                    batch = [({'tokens': words, 'positions1': positions1, 'positions2': positions2}, -1)]
                    with torch.no_grad():
                        tokens, positions1, positions2, lengths, y = my_collate(batch)

                        hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
                        outputs = m.forward(hidden_features)
                        _, pred = torch.max(outputs, 1)

                    relation_type = relation_vocab.lookup_id2str(pred.item())
                    if relation_type != '<unk>':
                        bioc_relation = bioc.BioCRelation()
                        passage.add_relation(bioc_relation)
                        bioc_relation.id = str(relation_id)
                        relation_id += 1
                        bioc_relation.infons['type'] = relation_type

                        node1 = bioc.BioCNode(former['id'], 'annotation 1')
                        bioc_relation.add_node(node1)
                        node2 = bioc.BioCNode(latter['id'], 'annotation 2')
                        bioc_relation.add_node(node2)

        with open(os.path.join(result_dumpdir, doc_name + ".bioc.xml"), 'w') as fp:
            bioc.dump(collection, fp)


def pretrain(train_token, train_entity, train_relation, test_token, test_entity, test_relation):
    word_alphabet, position_alphabet, relation_alphabet = dataset_stat(train_token, train_entity, train_relation)
    logging.info("training dataset stat completed")
    if opt.full_data:
        test_word_alphabet, test_position_alphabet, test_relation_alphabet = dataset_stat(test_token, test_entity, test_relation)
        word_alphabet = word_alphabet | test_word_alphabet
        position_alphabet = position_alphabet | test_position_alphabet
        relation_alphabet = relation_alphabet | test_relation_alphabet
        del test_word_alphabet, test_position_alphabet, test_relation_alphabet
        logging.info("test dataset stat completed")

    relation_vocab = vocab.Vocab(relation_alphabet, None, opt.relation_emb_size)
    word_vocab = vocab.Vocab(word_alphabet, opt.emb, opt.word_emb_size)
    position_vocab1 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)
    position_vocab2 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)
    logging.info("vocab build completed")

    logging.info("saving ... vocab")
    pickle.dump(word_vocab, open(os.path.join(opt.pretrain, 'word_vocab.pkl'), "wb"), True)
    pickle.dump(relation_vocab, open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), "wb"), True)
    pickle.dump(position_vocab1, open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), "wb"), True)
    pickle.dump(position_vocab2, open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), "wb"), True)

    # word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    # relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    # position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    # position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    # train_X, train_Y = utils.getRelatonInstance(train_token, train_entity, train_relation, word_vocab, relation_vocab, position_vocab1,
    #                                             position_vocab2)
    train_X, train_Y = utils.getRelationInstance1(train_token, train_entity, train_relation, word_vocab, relation_vocab, position_vocab1,
                                                position_vocab2)
    logging.info("training instance build completed, total {}".format(len(train_Y)))
    pickle.dump(train_X, open(os.path.join(opt.pretrain, 'train_X.pkl'), "wb"), True)
    pickle.dump(train_Y, open(os.path.join(opt.pretrain, 'train_Y.pkl'), "wb"), True)

    # test_X, test_Y = utils.getRelatonInstance(test_token, test_entity, test_relation, word_vocab, relation_vocab, position_vocab1,
    #                                           position_vocab2)
    test_X, test_Y = utils.getRelationInstance1(test_token, test_entity, test_relation, word_vocab, relation_vocab, position_vocab1,
                                              position_vocab2)
    logging.info("test instance build completed, total {}".format(len(test_Y)))
    pickle.dump(test_X, open(os.path.join(opt.pretrain, 'test_X.pkl'), "wb"), True)
    pickle.dump(test_Y, open(os.path.join(opt.pretrain, 'test_Y.pkl'), "wb"), True)


def train():

    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X = pickle.load(open(os.path.join(opt.pretrain, 'train_X.pkl'), 'rb'))
    train_Y = pickle.load(open(os.path.join(opt.pretrain, 'train_Y.pkl'), 'rb'))
    logging.info("training instance build completed, total {}".format(len(train_Y)))

    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate

    train_set = utils.RelationDataset(train_X, train_Y, opt.max_seq_len)
    # my_shuffle = False if opt.random_seed != 0 else True
    # train_loader = DataLoader(train_set, opt.batch_size, shuffle=my_shuffle, collate_fn=my_collate)
    train_loader = DataLoader(train_set, opt.batch_size, shuffle=True, collate_fn=my_collate)

    test_X = pickle.load(open(os.path.join(opt.pretrain, 'test_X.pkl'), 'rb'))
    test_Y = pickle.load(open(os.path.join(opt.pretrain, 'test_Y.pkl'), 'rb'))
    logging.info("test instance build completed, total {}".format(len(test_Y)))

    test_set = utils.RelationDataset(test_X, test_Y, opt.max_seq_len)
    test_loader = DataLoader(test_set, opt.batch_size, shuffle=False, collate_fn=my_collate)

    if opt.model.lower() == 'lstm':
        feature_extractor = LSTMFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                 opt.F_layers, opt.shared_hidden_size, opt.dropout)
    elif opt.model.lower() == 'cnn':
        feature_extractor = CNNFeatureExtractor(word_vocab, position_vocab1, position_vocab2,
                                                opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise RuntimeError('Unknown feature extractor {}'.format(opt.model))
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda(opt.gpu)

    if opt.model_high == 'capsule':
        # m = capsule.CapsuleNet(opt.shared_hidden_size, opt.dim_enlarge_rate, opt.init_dim_cap, relation_vocab)
        m = capsule.CapsuleNet(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'capsule_em':
        m = capsule_em.CapsuleNet_EM(opt.shared_hidden_size, relation_vocab)
    elif opt.model_high == 'mlp':
        m = baseline.MLP(opt.shared_hidden_size, relation_vocab)
    else:
        raise RuntimeError('Unknown model {}'.format(opt.model_high))
    if torch.cuda.is_available():
        m = m.cuda(opt.gpu)

    # m = capsule.CapsuleNet1(word_vocab, position_vocab1, position_vocab2, relation_vocab)

    iter_parameter = itertools.chain(*map(list, [feature_extractor.parameters(), m.parameters()]))
    # iter_parameter = itertools.chain(*map(list, [m.parameters()]))
    optimizer = optim.Adam(iter_parameter, lr=opt.learning_rate)
#    optimizer1 = optim.Adam(feature_extractor.parameters(), lr=opt.learning_rate)
    #   optimizer = optim.Adam(m.parameters(), lr=opt.learning_rate)

    num_iter = len(train_loader)
    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(opt.max_epoch):

        m.train()
        correct, total = 0, 0
        train_iter = iter(train_loader)
        for i in tqdm(range(num_iter)):

            tokens, positions1, positions2, lengths, targets = next(train_iter)

            hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
            outputs = m.forward(hidden_features)
            # outputs = m.forward(tokens, positions1, positions2)
            loss = m.loss(targets, outputs)

            optimizer.zero_grad()
#            optimizer1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(iter_parameter, opt.grad_clip)
            optimizer.step()
 #           optimizer1.step()

            total += tokens.size(0)
            _, pred = torch.max(outputs, 1)
            # if i==727:
            #     logging.info(pred)
            #     logging.info(targets)
            correct += (pred == targets).sum().item()


        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate(feature_extractor, m, test_loader)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(feature_extractor.state_dict(), '{}/feature_extractor.pth'.format(opt.output))
            torch.save(m.state_dict(), '{}/model.pth'.format(opt.output))
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def evaluate(feature_extractor, m, loader):
    with torch.no_grad():
        m.eval()
        it = iter(loader)
        correct = 0
        total = 0
        for tokens, positions1, positions2, lengths, targets in tqdm(it):

            hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
            outputs = m.forward(hidden_features)

            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

        acc = 100.0 * correct / total
        return acc

# def evaluate(m, loader):
#     with torch.no_grad():
#         m.eval()
#         it = iter(loader)
#         correct = 0
#         total = 0
#         for tokens, positions1, positions2, lengths, targets in tqdm(it):
#
#             outputs = m.forward(tokens, positions1, positions2, )
#
#             _, pred = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (pred == targets).sum().data.item()
#
#         acc = 100.0 * correct / total
#         return acc



