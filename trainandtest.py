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
import pickle
import sortedcontainers
import capsule

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





def train(train_token, train_entity, train_relation, test_token, test_entity, test_relation):

    word_alphabet, position_alphabet, relation_alphabet = dataset_stat(train_token, train_entity, train_relation)
    if opt.full_data:
        test_word_alphabet, test_position_alphabet, test_relation_alphabet = dataset_stat(test_token, test_entity, test_relation)
        word_alphabet = word_alphabet | test_word_alphabet
        position_alphabet = position_alphabet | test_position_alphabet
        relation_alphabet = relation_alphabet | test_relation_alphabet
        del test_word_alphabet, test_position_alphabet, test_relation_alphabet

    relation_vocab = vocab.Vocab(relation_alphabet, None, opt.relation_emb_size)
    word_vocab = vocab.Vocab(word_alphabet, opt.emb, opt.word_emb_size)
    position_vocab1 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)
    position_vocab2 = vocab.Vocab(position_alphabet, None, opt.position_emb_size)


    my_collate = utils.unsorted_collate
    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X, train_Y = utils.getRelatonInstance('train', train_token, train_entity, train_relation, word_vocab, relation_vocab, position_vocab1,
                                                position_vocab2)
    train_set = utils.RelationDataset(train_X, train_Y, opt.max_seq_len)
    my_shuffle = False if opt.random_seed != 0 else True
    train_loader = DataLoader(train_set, opt.batch_size, shuffle=my_shuffle, collate_fn = my_collate)


    test_X, test_Y = utils.getRelatonInstance('test', test_token, test_entity, test_relation, word_vocab, relation_vocab, position_vocab1,
                                              position_vocab2)
    test_set = utils.RelationDataset(test_X, test_Y, opt.max_seq_len)
    test_loader = DataLoader(test_set, opt.batch_size, shuffle=False, collate_fn=my_collate)

    if opt.model == 1:
        m = capsule.CapsuleNet(word_vocab, position_vocab1, position_vocab2, relation_vocab)
    else:
        m = baseline.CNN(word_vocab, position_vocab1, position_vocab2, relation_vocab)
    if torch.cuda.is_available():
        m.cuda()

    optimizer = optim.Adam(m.parameters(), lr=opt.learning_rate)

    num_iter = len(train_loader)
    best_acc = 0.0
    for epoch in range(opt.max_epoch):

        m.train()
        correct, total = 0, 0
        train_iter = iter(train_loader)
        for i in tqdm(range(num_iter)):

            tokens, positions1, positions2, targets = next(train_iter)

            outputs = m.forward(tokens, positions1, positions2)
            loss = m.loss(targets, outputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), opt.grad_clip)
            optimizer.step()

            total += tokens.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()


        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            with open(os.path.join(opt.output, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(m.state_dict(), '{}/model.pth'.format(opt.output))
            logging.info('New best accuracy: {}'.format(best_acc))


def evaluate(m, loader):
    with torch.no_grad():
        m.eval()
        it = iter(loader)
        correct = 0
        total = 0
        for tokens, positions1, positions2, targets in tqdm(it):
            outputs = m.forward(tokens, positions1, positions2)

            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

        acc = 100.0 * correct / total
        return acc


