class CNN(nn.Module):
    def __init__(self, word_vocab, position1_vocab, position2_vocab, relation_vocab):
        super(CNN, self).__init__()

        self.word_emb = nn.Embedding(word_vocab.vocab_size, word_vocab.emb_size, padding_idx=word_vocab.pad_idx)
        self.word_emb.weight.data = torch.from_numpy(word_vocab.embeddings).float()

        self.position1_emb = nn.Embedding(position1_vocab.vocab_size, position1_vocab.emb_size,
                                          padding_idx=position1_vocab.pad_idx)
        self.position1_emb.weight.data = torch.from_numpy(position1_vocab.embeddings).float()

        self.position2_emb = nn.Embedding(position2_vocab.vocab_size, position2_vocab.emb_size,
                                          padding_idx=position2_vocab.pad_idx)
        self.position2_emb.weight.data = torch.from_numpy(position2_vocab.embeddings).float()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=100,
            kernel_size=(
            3, self.word_emb.embedding_dim + self.position1_emb.embedding_dim + self.position2_emb.embedding_dim),
            stride=1,
            padding=0,
        )

        self.out = nn.Linear(100, relation_vocab.vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, positions1, positions2):  # (bz, seq)
        bz, seq = tokens.size()

        tokens = self.word_emb(tokens)  # (bz, seq, emb)
        positions1 = self.position1_emb(positions1)
        positions2 = self.position2_emb(positions2)

        x = torch.cat((tokens, positions1, positions2), 2).unsqueeze(1)  # (bz, 1, seq, word_emb+pos1_emb+pos2_emb)

        x = F.relu(self.conv1(x).squeeze(-1))
        x = F.max_pool1d(x, seq - 2).squeeze(-1)

        x = self.out(x)
        return x

    def loss(self, gold, pred):
        cost = self.criterion(pred, gold)

        return cost


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
        m = capsule.CapsuleNet(opt.shared_hidden_size, relation_vocab, entity_type_vocab, entity_vocab)
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

        for i in tqdm(range(len(test_relation))):  # this procedure should keep consistent with utils.getRelatonInstance

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

            for _, relation in doc_relation.iterrows():

                # find entity mention
                entity1 = doc_entity[(doc_entity['id'] == relation['entity1_id'])].iloc[0]
                entity2 = doc_entity[(doc_entity['id'] == relation['entity2_id'])].iloc[0]
                # find all sentences between entity1 and entity2
                former = entity1 if entity1['start'] < entity2['start'] else entity2
                latter = entity2 if entity1['start'] < entity2['start'] else entity1
                context_token = doc_token[
                    (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
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
                    logging.debug(
                        'former_head not found, entity {} {} {}'.format(former['id'], former['start'], former['text']))
                    continue
                if latter_head == -1:
                    logging.debug(
                        'latter_head not found, entity {} {} {}'.format(latter['id'], latter['start'], latter['text']))
                    continue

                i = 0
                for _, token in context_token.iterrows():
                    word = utils.normalizeWord(token['text'])
                    words.append(word_vocab.lookup(word))

                    positions1.append(position_vocab1.lookup(former_head - i))
                    positions2.append(position_vocab2.lookup(latter_head - i))

                    i += 1

                # here we ignore utils.RelationDataset(test_X, test_Y, opt.max_seq_len)
                # [({'tokens': [171, 35, 371, 304, 6, 243, 389, 106, 2],
                #    'positions2': [107, 106, 105, 104, 103, 102, 101, 100, 99],
                #    'positions1': [105, 104, 103, 102, 101, 100, 99, 98, 97]}, 3), []]

                batch = [({'tokens': words, 'positions1': positions1, 'positions2': positions2}, -1)]
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

            with open(os.path.join(result_dumpdir, doc_name + ".bioc.xml"), 'w') as fp:
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
        m = capsule.CapsuleNet(opt.shared_hidden_size, relation_vocab, entity_type_vocab, entity_vocab)
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

                    former = doc_entity.iloc[former_idx]
                    latter = doc_entity.iloc[latter_idx]

                    if math.fabs(latter['sent_idx'] - former['sent_idx']) >= opt.sent_window:
                        continue

                    type_constraint = utils.relationConstraint(former['type'], latter['type'])
                    if type_constraint == 0:
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



# enumerate all entity pairs
def getRelationInstance1(tokens, entities, relations, names, word_vocab, relation_vocab, position_vocab1, position_vocab2):
    X = []
    Y = []
    other = [] # other is used for outputing results, it's usually used for test set
    cnt_neg = 0

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i] # entity are sorted by start offset
        doc_name = names[i]

        row_num = doc_entity.shape[0]

        for latter_idx in range(row_num):

            for former_idx in range(row_num):

                if former_idx < latter_idx:

                    former = doc_entity.iloc[former_idx]
                    latter = doc_entity.iloc[latter_idx]
                    # if former['id']=='108237':
                    #     pass

                    if math.fabs(latter['sent_idx']-former['sent_idx']) >= opt.sent_window:
                        continue

                    type_constraint = relationConstraint(former['type'], latter['type'])
                    if type_constraint == 0:
                        continue

                    gold_relations = doc_relation[
                        (
                                ((doc_relation['entity1_id'] == former['id']) & (
                                            doc_relation['entity2_id'] == latter['id']))
                                |
                                ((doc_relation['entity1_id'] == latter['id']) & (
                                            doc_relation['entity2_id'] == former['id']))
                        )
                    ]
                    if gold_relations.shape[0] > 1:
                        raise RuntimeError("the same entity pair has more than one relations")



                    context_token = doc_token[
                        (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]
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
                        word = normalizeWord(token['text'])
                        words.append(word_vocab.lookup(word))

                        positions1.append(position_vocab1.lookup(former_head - i))
                        positions2.append(position_vocab2.lookup(latter_head - i))

                        i += 1

                    X.append({'tokens': words, 'positions1': positions1, 'positions2': positions2})
                    if gold_relations.shape[0] == 0:
                        Y.append(relation_vocab.lookup('<unk>'))
                        cnt_neg += 1
                    else:
                        Y.append(relation_vocab.lookup(gold_relations.iloc[0]['type']))

                    other_info = {}
                    other_info['doc_name'] = doc_name
                    other_info['former_id'] = former['id']
                    other_info['latter_id'] = latter['id']
                    other.append(other_info)




    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y, other

# truncate after feature
def getRelationInstance3(tokens, entities, relations, names, word_vocab, relation_vocab, entity_type_vocab, entity_vocab, position_vocab1, position_vocab2):
    X = []
    Y = []
    other = [] # other is used for outputing results, it's usually used for test set
    cnt_neg = 0

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i] # entity are sorted by start offset
        doc_name = names[i]


        row_num = doc_entity.shape[0]

        for latter_idx in range(row_num):

            for former_idx in range(row_num):

                if former_idx < latter_idx:

                    former = doc_entity.iloc[former_idx]
                    latter = doc_entity.iloc[latter_idx]

                    if math.fabs(latter['sent_idx']-former['sent_idx']) >= opt.sent_window:
                        continue

                    type_constraint = relationConstraint(former['type'], latter['type'])
                    if type_constraint == 0:
                        continue

                    gold_relations = doc_relation[
                        (
                                ((doc_relation['entity1_id'] == former['id']) & (
                                            doc_relation['entity2_id'] == latter['id']))
                                |
                                ((doc_relation['entity1_id'] == latter['id']) & (
                                            doc_relation['entity2_id'] == former['id']))
                        )
                    ]
                    if gold_relations.shape[0] > 1:
                        raise RuntimeError("the same entity pair has more than one relations")

                    # here we retrieve all the sentences inbetween two entities, sentence of former, sentence ..., sentence of latter
                    sent_idx = former['sent_idx']
                    context_token = pd.DataFrame(columns=doc_token.columns)
                    base = 0
                    former_tf_start, former_tf_end = -1, -1
                    latter_tf_start, latter_tf_end = -1, -1
                    while sent_idx <= latter['sent_idx']:
                        sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]

                        if former['sent_idx'] == sent_idx:
                            former_tf_start, former_tf_end = base+former['tf_start'], base+former['tf_end']
                        if latter['sent_idx'] == sent_idx:
                            latter_tf_start, latter_tf_end = base+latter['tf_start'], base+latter['tf_end']

                        context_token = context_token.append(sentence, ignore_index=True)

                        base += len(sentence['text'])
                        sent_idx += 1


                    words = []
                    positions1 = []
                    positions2 = []
                    former_token = []
                    latter_token = []
                    i = 0
                    for _, token in context_token.iterrows():
                        word = normalizeWord(token['text'])
                        words.append(word_vocab.lookup(word))

                        if i < former_tf_start:
                            positions1.append(position_vocab1.lookup(former_tf_start - i))
                        elif i > former_tf_end:
                            positions1.append(position_vocab1.lookup(former_tf_end - i))
                        else:
                            positions1.append(position_vocab1.lookup(0))
                            former_token.append(entity_vocab.lookup(word))

                        if i < latter_tf_start:
                            positions2.append(position_vocab2.lookup(latter_tf_start - i))
                        elif i > latter_tf_end:
                            positions2.append(position_vocab2.lookup(latter_tf_end - i))
                        else:
                            positions2.append(position_vocab2.lookup(0))
                            latter_token.append(entity_vocab.lookup(word))

                        i += 1


                    assert len(former_token)>0
                    assert len(latter_token)>0



                    if len(words) > opt.max_seq_len:
                        # truncate
                        logging.debug("exceed max_seq_len {} {}".format(doc_name, len(words)))
                        words = words[:opt.max_seq_len]
                        positions1 = positions1[:opt.max_seq_len]
                        positions2 = positions2[:opt.max_seq_len]


                    features = {'tokens': words, 'positions1': positions1, 'positions2': positions2}
                    if type_constraint == 1:
                        features['e1_type'] = entity_type_vocab.lookup(former['type'])
                        features['e2_type'] = entity_type_vocab.lookup(latter['type'])
                        features['e1_token'] = former_token
                        features['e2_token'] = latter_token
                    else:
                        features['e1_type'] = entity_type_vocab.lookup(latter['type'])
                        features['e2_type'] = entity_type_vocab.lookup(former['type'])
                        features['e1_token'] = latter_token
                        features['e2_token'] = former_token

                    X.append(features)

                    if gold_relations.shape[0] == 0:
                        Y.append(relation_vocab.lookup('<unk>'))
                        cnt_neg += 1
                    else:
                        Y.append(relation_vocab.lookup(gold_relations.iloc[0]['type']))

                    other_info = {}
                    other_info['doc_name'] = doc_name
                    other_info['former_id'] = former['id']
                    other_info['latter_id'] = latter['id']
                    other.append(other_info)




    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y, other


def getRelatonInstance(tokens, entities, relations, word_vocab, relation_vocab, position_vocab1, position_vocab2):

    X = []
    Y = []

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i]

        for index, relation in doc_relation.iterrows():

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

                word = normalizeWord(token['text'])
                words.append(word_vocab.lookup(word))

                positions1.append(position_vocab1.lookup(former_head-i))
                positions2.append(position_vocab2.lookup(latter_head-i))

                i += 1


            X.append({'tokens': words, 'positions1': positions1, 'positions2':positions2})
            Y.append(relation_vocab.lookup(relation['type']))


    return X, Y


class SentimentClassifier(nn.Module):
    def __init__(self, context_feature_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab, et_num_vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'

        self.entity_type_emb = nn.Embedding(entity_type_vocab.vocab_size, entity_type_vocab.emb_size, padding_idx=entity_type_vocab.pad_idx)
        self.entity_type_emb.weight.data = torch.from_numpy(entity_type_vocab.embeddings).float()

        self.entity_emb = nn.Embedding(entity_vocab.vocab_size, entity_vocab.emb_size, padding_idx=entity_vocab.pad_idx)
        self.entity_emb.weight.data = torch.from_numpy(entity_vocab.embeddings).float()

        self.tok_num_betw_emb = nn.Embedding(tok_num_betw_vocab.vocab_size, tok_num_betw_vocab.emb_size, padding_idx=tok_num_betw_vocab.pad_idx)
        self.tok_num_betw_emb.weight.data = torch.from_numpy(tok_num_betw_vocab.embeddings).float()

        self.et_num_emb = nn.Embedding(et_num_vocab.vocab_size, et_num_vocab.emb_size, padding_idx=et_num_vocab.pad_idx)
        self.et_num_emb.weight.data = torch.from_numpy(et_num_vocab.embeddings).float()

        self.dot_att = feature_extractor.DotAttentionLayer(entity_vocab.emb_size)

        self.criterion = nn.CrossEntropyLoss()

        self.input_size = context_feature_size+2*entity_type_vocab.emb_size+2*entity_vocab.emb_size + \
                          tok_num_betw_vocab.emb_size + et_num_vocab.emb_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(self.input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, relation_vocab.vocab_size))
        #self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, hidden_features, x2, x1):
        tokens, positions1, positions2, e1_token, e2_token = x2
        e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths = x1

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        return self.net(x)

    def loss(self, by, y_pred):

        return self.criterion(y_pred, by)


class WordVocab:
    def __init__(self, txt_file):
        with open(txt_file, 'r') as inf:
            parts = inf.readline().split()
            assert len(parts) == 2
            self.vocab_size, self.emb_size = int(parts[0]), int(parts[1])
            opt.vocab_size = self.vocab_size
            opt.emb_size = self.emb_size
            # add an UNK token
            self.unk_tok = '<unk>'
            self.unk_idx = 0
            self.vocab_size += 1
            self.v2wvocab = ['<unk>']
            self.w2vvocab = {'<unk>': 0}
            self.embeddings = np.empty((self.vocab_size, self.emb_size), dtype=np.float)
            cnt = 1
            for line in inf.readlines():
                parts = line.rstrip().split(' ')
                word = parts[0]
                # add to vocab
                self.v2wvocab.append(word)
                self.w2vvocab[word] = cnt
                # load vector
                vector = [float(x) for x in parts[-self.emb_size:]]
                self.embeddings[cnt] = vector
                cnt += 1

        self.eos_tok = '</s>'
        opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
        # randomly initialize <unk> vector
        self.embeddings[self.unk_idx] = np.random.normal(0, 1, size=self.emb_size)
        # normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1, 1)
        # zero </s>
        self.embeddings[self.eos_idx] = 0

    def init_embed_layer(self):
        word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.eos_idx)
        if not opt.random_emb:
            word_emb.weight.data = torch.from_numpy(self.embeddings).float()
        return word_emb

    def lookup(self, word):

        if word in self.w2vvocab:
            return self.w2vvocab[word]
        return self.unk_idx


class RelationVocab:

    def __init__(self, relations):

        relationName = sortedcontainers.SortedSet()
        for relation in relations:
            relationName.update(relation['type'].tolist())

        self.id2str = list(relationName)
        self.vocab_size = len(self.id2str)
        self.str2id = {}
        cnt = 0
        for str in self.id2str:
            self.str2id[str] = cnt
            cnt += 1
        # add an UNK relation
        self.unk = '<unk>'
        self.unk_idx = self.vocab_size
        self.id2str.append(self.unk)
        self.str2id[self.unk] = self.unk_idx
        self.vocab_size += 1

    def lookup(self, item):

        if item in self.str2id:
            return self.str2id[item]
        return self.unk_idx


def relationConstraint_chapman1(relation_type, type1, type2):

    if relation_type=='do':
        if (type1 == 'Drug' and type2 == 'Dose') or (type1 == 'Dose' and type2 == 'Drug'):
            return True
        else:
            return False

    elif relation_type=='fr':
        if (type1 == 'Drug' and type2 == 'Frequency') or (type1 == 'Frequency' and type2 == 'Drug'):
            return True
        else:
            return False
    elif relation_type=='manner/route':
        if (type1 == 'Drug' and type2 == 'Route') or (type1 == 'Route' and type2 == 'Drug'):
            return True
        else:
            return False
    elif relation_type=='Drug_By Patient':
        if (type1 == 'Drug By' and type2 == 'Patient') or (type1 == 'Patient' and type2 == 'Drug By'):
            return True
        else:
            return False
    elif relation_type=='severity_type':
        if (type1 == 'Indication' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'Indication') or \
                (type1 == 'ADE' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'ADE') or \
                (type1 == 'SSLIF' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'SSLIF'):
            return True
        else:
            return False
    elif relation_type=='adverse':
        if (type1 == 'Drug' and type2 == 'ADE') or (type1 == 'ADE' and type2 == 'Drug'):
            return True
        else:
            return False
    elif relation_type=='reason':
        if (type1 == 'Drug' and type2 == 'Indication') or (type1 == 'Indication' and type2 == 'Drug'):
            return True
        else:
            return False
    elif relation_type=='Drug_By Physician':
        if (type1 == 'Drug By' and type2 == 'Physician') or (type1 == 'Physician' and type2 == 'Drug By'):
            return True
        else:
            return False
    elif relation_type=='du':
        if (type1 == 'Drug' and type2 == 'Duration') or (type1 == 'Duration' and type2 == 'Drug'):
            return True
        else:
            return False
    else:
        raise RuntimeError("unknown relation type")


def other_stat():
    opt.otherdata = "/Users/feili/Desktop/umass/bioC_data/other"

    # preprocess.preprocess(data_path)

    # train_token, train_entity, train_relation, train_name = preprocess.loadPreprocessData(data_path)

    other_token, other_entity, other_relation, other_name = {}, {}, {}, {}
    for other_dir in os.listdir(opt.otherdata):
        other_token[other_dir], other_entity[other_dir], other_relation[other_dir], other_name[
            other_dir] = preprocess.loadPreprocessData(os.path.join(opt.otherdata, other_dir))
    #
    # for other in other_token:
    #     other_word_alphabet, other_postag_alphabet, other_relation_alphabet, other_entity_type_alphabet, other_entity_alphabet = dataset_stat(
    #         other_token[other], other_entity[other], other_relation[other])

    relation_argument = set()

    for domain in other_relation:
        relations = other_relation[domain]
        entities = other_entity[domain]
        for i, doc_relation in enumerate(relations):
            doc_entity = entities[i]

            for _, r in doc_relation.iterrows():
                entity1 = doc_entity[(doc_entity['id'] == r['entity1_id'])].iloc[0]
                entity2 = doc_entity[(doc_entity['id'] == r['entity2_id'])].iloc[0]
                rt = r['type']
                e1 = entity1['type']
                e2 = entity2['type']
                relation_argument.add("{} | {} | {}".format(rt, e1, e2))

    print relation_argument

def entity_type_stat():
    opt.traindata = "/Users/feili/Desktop/umass/MADE/MADE-1.0"
    opt.testdata = "/Users/feili/Desktop/umass/MADE/made_test_data"

    entity_type = set()
    train_token, train_entity, _, train_name = preprocess.loadPreprocessData(opt.testdata)
    for doc_entity in train_entity:

        for _, entity in doc_entity.iterrows():
            entity_type.add(entity['type'])

    print(entity_type)