import utils
import vocab
from torch.utils.data import DataLoader
from options import opt
from tqdm import tqdm
import logging

def train(args, train_token, train_entity, train_relation, test_token, test_entity, test_relation):

    relation_vocab = vocab.RelationVocab(train_relation)

    word_vocab = vocab.WordVocab(args.emb)

    my_collate = utils.unsorted_collate
    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X, train_Y = utils.getRelatonInstance(train_token, train_entity, train_relation, word_vocab, relation_vocab)
    train_set = utils.RelationDataset(train_X, train_Y, opt.max_seq_len)
    my_shuffle = False if opt.random_seed != 0 else True
    train_loader = DataLoader(train_set, opt.batch_size, shuffle=my_shuffle, collate_fn = my_collate)


    test_X, test_Y = utils.getRelatonInstance(test_token, test_entity, test_relation, word_vocab, relation_vocab)
    test_set = utils.RelationDataset(test_X, test_Y, opt.max_seq_len)
    test_loader = DataLoader(test_set, opt.batch_size, shuffle=False, collate_fn=my_collate)



    num_iter = len(train_loader)

    for epoch in range(opt.max_epoch):
        logging.info('epoch {}'.format(epoch))

        train_iter = iter(train_loader)
        for i in tqdm(range(num_iter)):

            inputs, targets = next(train_iter)
            pass


