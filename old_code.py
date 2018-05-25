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


