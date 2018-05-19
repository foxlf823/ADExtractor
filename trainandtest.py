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
import numpy as np
import random
import copy

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



# used when not enumerate all entities
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

# used when enumerate all entities
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

                    if utils.relationConstraint(former['type'], latter['type']) == False:
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
                    if relation_type == '<unk>':
                        continue
                    elif utils.relationConstraint1(relation_type, former['type'], latter['type']) == False:
                        continue
                    else:
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

# used when entities have been enumerated, just translate into bioc format
def test2(test_token, test_entity, test_relation, test_name, result_dumpdir):
    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    logging.info("loading ... result")
    results = pickle.load(open(os.path.join(opt.output, 'results.pkl'), "rb"))

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

        relation_id = 1
        for result in results:

            if doc_name == result['doc_name'] :

                former = doc_entity[ (doc_entity['id'] == result['former_id'])].iloc[0]
                latter = doc_entity[(doc_entity['id'] == result['latter_id'])].iloc[0]

                relation_type = relation_vocab.lookup_id2str(result['type'])
                if relation_type == '<unk>':
                    continue
                elif utils.relationConstraint1(relation_type, former['type'], latter['type']) == False:
                    continue
                else:
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

def pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation, test_name):
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
    train_X, train_Y, _ = utils.getRelationInstance1(train_token, train_entity, train_relation, train_name, word_vocab, relation_vocab, position_vocab1,
                                                position_vocab2)
    logging.info("training instance build completed, total {}".format(len(train_Y)))
    pickle.dump(train_X, open(os.path.join(opt.pretrain, 'train_X.pkl'), "wb"), True)
    pickle.dump(train_Y, open(os.path.join(opt.pretrain, 'train_Y.pkl'), "wb"), True)

    # test_X, test_Y = utils.getRelatonInstance(test_token, test_entity, test_relation, word_vocab, relation_vocab, position_vocab1,
    #                                           position_vocab2)
    test_X, test_Y, test_other = utils.getRelationInstance1(test_token, test_entity, test_relation, test_name, word_vocab, relation_vocab, position_vocab1,
                                              position_vocab2)
    logging.info("test instance build completed, total {}".format(len(test_Y)))
    pickle.dump(test_X, open(os.path.join(opt.pretrain, 'test_X.pkl'), "wb"), True)
    pickle.dump(test_Y, open(os.path.join(opt.pretrain, 'test_Y.pkl'), "wb"), True)
    pickle.dump(test_other, open(os.path.join(opt.pretrain, 'test_Other.pkl'), "wb"), True)


def makeDatasetForEachClass(train_X, train_Y, relation_vocab, my_collate):
    train_X_classified = {}
    train_Y_classified = {}
    for i in range(len(train_X)):
        x = train_X[i]
        y = train_Y[i]
        class_name = relation_vocab.lookup_id2str(y)
        if class_name in train_X_classified.keys():
            train_X_classified[class_name].append(x)
            train_Y_classified[class_name].append(y)
        else:
            train_X_classified[class_name] = [x]
            train_Y_classified[class_name] = [y]


    train_sets = [] # each class corresponds to a set, loader, sample
    train_loaders = []
    train_samples = []
    train_numbers = []
    train_iters = []
    for class_name in train_Y_classified:
        x = train_X_classified[class_name]
        y = train_Y_classified[class_name]
        train_numbers.append((class_name, len(y)))

        train_set = utils.RelationDataset(x, y, opt.max_seq_len)
        train_sets.append(train_set)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
        train_samples.append(train_sampler)
        train_loader = DataLoader(train_set, 1, shuffle=False, sampler=train_sampler, collate_fn=my_collate)
        train_loaders.append(train_loader)
        train_iter = iter(train_loader)
        train_iters.append(train_iter)

    return train_loaders, train_iters, train_numbers

def makeDatasetWithoutUnknown(test_X, test_Y, relation_vocab, b_shuffle, my_collate):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y != relation_vocab.unk_idx:
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk, opt.max_seq_len)
    test_loader = DataLoader(test_set, opt.batch_size, shuffle=b_shuffle, collate_fn=my_collate)
    it = iter(test_loader)
    logging.info("instance after removing unknown, {}".format(len(test_Y_remove_unk)))
    return test_loader, it

def randomSampler(dataset_list, ratio):
    a = range(len(dataset_list))
    random.shuffle(a)
    indices = a[:int(len(dataset_list)*ratio)]
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return sampler

def makeDatasetUnknown(test_X, test_Y, relation_vocab, my_collate, ratio):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y == relation_vocab.unk_idx:
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk, opt.max_seq_len)

    test_loader = DataLoader(test_set, opt.batch_size, shuffle=False, sampler=randomSampler(test_Y_remove_unk, ratio), collate_fn=my_collate)
    it = iter(test_loader)

    return test_loader, it

def train():

    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X = pickle.load(open(os.path.join(opt.pretrain, 'train_X.pkl'), 'rb'))
    train_Y = pickle.load(open(os.path.join(opt.pretrain, 'train_Y.pkl'), 'rb'))
    logging.info("total training instance {}".format(len(train_Y)))

    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate


    if opt.strategy == 'all':
        train_loader = DataLoader(utils.RelationDataset(train_X, train_Y, opt.max_seq_len),
                                  opt.batch_size, shuffle=True, collate_fn=my_collate)
        train_iter = iter(train_loader)
        num_iter = len(train_loader)
    elif opt.strategy == 'no-unk':
        train_loader, train_iter = makeDatasetWithoutUnknown(train_X, train_Y, relation_vocab, True, my_collate)
        num_iter = len(train_loader)
    elif opt.strategy == 'balance':
        train_loaders, train_iters, train_numbers = makeDatasetForEachClass(train_X, train_Y, relation_vocab, my_collate)
        for t in train_numbers:
            logging.info(t)
        # use the median number of instance number of all classes except unknown
        num_iter = int(
            np.median(np.array([num for class_name, num in train_numbers if class_name != relation_vocab.unk_tok])))
    elif opt.strategy == 'part-unk':
        train_loader, train_iter = makeDatasetWithoutUnknown(train_X, train_Y, relation_vocab, True, my_collate)
        num_iter = len(train_loader)
        unk_loader, unk_iter = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, opt.unk_ratio)
    else:
        raise RuntimeError("unsupport training strategy")

    test_X = pickle.load(open(os.path.join(opt.pretrain, 'test_X.pkl'), 'rb'))
    test_Y = pickle.load(open(os.path.join(opt.pretrain, 'test_Y.pkl'), 'rb'))
    test_Other = pickle.load(open(os.path.join(opt.pretrain, 'test_Other.pkl'), 'rb'))
    logging.info("total test instance {}".format(len(test_Y)))
    #test_loader, _ = makeDatasetWithoutUnknown(test_X, test_Y, relation_vocab, False, my_collate)
    test_loader = DataLoader(utils.RelationDataset(test_X, test_Y, opt.max_seq_len),
                              opt.batch_size, shuffle=False, collate_fn=my_collate) # drop_last=True


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

    iter_parameter = itertools.chain(*map(list, [feature_extractor.parameters(), m.parameters()]))
    optimizer = optim.Adam(iter_parameter, lr=opt.learning_rate)

    if opt.tune_wordemb == False:
        utils.freeze_layer(feature_extractor.word_emb)
        #feature_extractor.word_emb.weight.requires_grad = False

    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(opt.max_epoch):

        m.train()
        correct, total = 0, 0

        for i in tqdm(range(num_iter)):

            if opt.strategy == 'all' or opt.strategy == 'no-unk' or opt.strategy == 'part-unk':
                tokens, positions1, positions2, lengths, targets = utils.endless_get_next_batch_without_rebatch(train_loader, train_iter)

            elif opt.strategy == 'balance':
                tokens, positions1, positions2, lengths, targets = utils.endless_get_next_batch(train_loaders,
                                                                                                train_iters)
            else:
                raise RuntimeError("unsupport training strategy")



            hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
            outputs = m.forward(hidden_features)
            loss = m.loss(targets, outputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(iter_parameter, opt.grad_clip)
            optimizer.step()

            total += tokens.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()

            if opt.strategy == 'part-unk':
                tokens, positions1, positions2, lengths, targets = utils.endless_get_next_batch_without_rebatch(
                    unk_loader, unk_iter)
                hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
                outputs = m.forward(hidden_features)
                loss = m.loss(targets, outputs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(iter_parameter, opt.grad_clip)
                optimizer.step()

        if opt.strategy == 'part-unk':
            unk_loader, unk_iter = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, opt.unk_ratio)

        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate(feature_extractor, m, test_loader, test_Other)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(feature_extractor.state_dict(), '{}/feature_extractor.pth'.format(opt.output))
            torch.save(m.state_dict(), '{}/model.pth'.format(opt.output))
            pickle.dump(test_Other, open(os.path.join(opt.output, 'results.pkl'), "wb"), True)
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def evaluate(feature_extractor, m, loader, other):
    #results = []
    m.eval()
    it = iter(loader)
    start, end = 0, 0
    correct = 0
    total = 0
    for tokens, positions1, positions2, lengths, targets in it:

        if tokens.size(0) == 1 and (opt.model_bn or opt.model_high_bn):
            tokens = tokens.expand(2, -1)
            positions1 = positions1.expand(2, -1)
            positions2 = positions2.expand(2, -1)
            lengths = lengths.expand(2)
            targets = targets.expand(2)

        with torch.no_grad():
            hidden_features = feature_extractor.forward(tokens, positions1, positions2, lengths)
            outputs = m.forward(hidden_features)

            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

        start = end
        end = end + tokens.size(0)

        # batch_other = copy.deepcopy(other[start:end])
        # for i, d in enumerate(batch_other):
        #     d["type"] = pred[i].item()
        # results.extend(batch_other)

        for i, d in enumerate(other[start:end]):
            d["type"] = pred[i].item()

    acc = 100.0 * correct / total
    return acc




