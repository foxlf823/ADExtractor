import logging
import cPickle as pickle
import os
from options import opt
import preprocess
import trainandtest
import pandas as pd
import shutil

def printTrainingInstanceAsSequence():
    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(opt.pretrain, 'word_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(opt.pretrain, 'relation_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(opt.pretrain, 'position_vocab2.pkl'), 'rb'))

    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X = pickle.load(open(os.path.join(opt.pretrain, 'train_X.pkl'), 'rb'))
    train_Y = pickle.load(open(os.path.join(opt.pretrain, 'train_Y.pkl'), 'rb'))
    logging.info("total training instance {}".format(len(train_Y)))

    for i, x in enumerate(train_X):
        y = train_Y[i]
        relation_str = relation_vocab.lookup_id2str(y)
        if relation_str == "adverse":
            sequence = ""
            for j, token_id in enumerate(x['tokens']):

                p1, p2 = position_vocab1.lookup_id2str(x['positions1'][j]), position_vocab2.lookup_id2str(x['positions2'][j])
                token = word_vocab.lookup_id2str(token_id)
                if p1 == 0 or p2 == 0:
                    sequence += "["+token+"] "
                else:
                    sequence += token + " "
            print(sequence)

def errorAnalysisForRelation(type, goldDir, result, vocab_dir, error_dir):
    if os.path.exists(error_dir):
            shutil.rmtree(error_dir)
            os.makedirs(error_dir)
    else:
        os.makedirs(error_dir)

    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(goldDir)

    logging.info("loading ... vocab")
    relation_vocab = pickle.load(open(os.path.join(vocab_dir, 'relation_vocab.pkl'),  'rb'))
    #word_vocab = pickle.load(open(os.path.joint(vocab_dir, 'word_vocab.pkl'), 'rb'))

    logging.info("loading ... result")
    results = pickle.load(open(result, "rb"))

    for i in range(len(test_relation)):

        doc_entity = test_entity[i]
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_relation = test_relation[i]

        # print('########### '+doc_name)
        # print('########### FP')
        listFP = []
        for result in results:

            if doc_name == result['doc_name'] :

                relation_type = relation_vocab.lookup_id2str(result['type'])
                if relation_type == type:

                    gold_relations = doc_relation[
                        (
                            (((doc_relation['entity1_id'] == result['former_id']) & (doc_relation['entity2_id'] == result['latter_id']))
                                |
                                ((doc_relation['entity1_id'] == result['latter_id']) & (doc_relation['entity2_id'] == result['former_id'])))
                            &
                            (doc_relation['type'] == relation_type)
                        )
                    ]
                    if gold_relations.shape[0] != 0 :
                        continue


                    former = doc_entity[(doc_entity['id'] == result['former_id'])].iloc[0]
                    latter = doc_entity[(doc_entity['id'] == result['latter_id'])].iloc[0]

                    context_token = doc_token[
                        (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]

                    # print("{}: {} | {}: {}".format(former['id'], former['text'], latter['id'], latter['text']))
                    sequence = ""
                    for _, token in context_token.iterrows():
                        if token['start'] == former['start'] or token['start'] == latter['start']:
                            sequence += "["
                        sequence += token['text']
                        if token['end'] == former['end'] or token['end'] == latter['end']:
                            sequence += "]"
                        sequence += " "
                    # print(sequence)
                    listFP.append("{} {} {} | {} {} {} \n{}\n".format(former['id'], former['type'], former['text'], latter['id'], latter['type'], latter['text'], sequence))


        # print('########### FN')
        listFN = []
        for _, relation in doc_relation.iterrows():
            if relation['type'] != type:
                continue

            find = False
            for result in results:

                relation_type = relation_vocab.lookup_id2str(result['type'])

                if doc_name == result['doc_name'] and relation_type==relation['type'] and \
                        ((relation['entity1_id']==result['former_id'] and relation['entity2_id'] == result['latter_id']) \
                        or
                        (relation['entity1_id'] == result['latter_id'] and relation['entity2_id'] == result['former_id'])):
                    find = True
                    break

            if not find:


                entity1 = doc_entity[(doc_entity['id'] == relation['entity1_id'])].iloc[0]
                entity2 = doc_entity[(doc_entity['id'] == relation['entity2_id'])].iloc[0]

                former = entity1 if entity1['start'] < entity2['start'] else entity2
                latter = entity2 if entity1['start'] < entity2['start'] else entity1

                context_token = doc_token[
                    (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]

                # print("{}: {} | {}: {}".format(former['id'], former['text'], latter['id'], latter['text']))
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == former['start'] or token['start'] == latter['start']:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == former['end'] or token['end'] == latter['end']:
                        sequence += "]"
                    sequence += " "
                # print(sequence)
                listFN.append(
                    "{} {} {} | {} {} {} \n{}\n".format(former['id'], former['type'], former['text'], latter['id'],
                                                        latter['type'], latter['text'], sequence))

        if len(listFP)!= 0 or len(listFN)!= 0:
            with open(os.path.join(error_dir, doc_name+".txt"), 'w') as f:
                if len(listFP) != 0:
                    f.write("########### FP\n")
                for l in listFP:
                    f.write(l)
                f.write("\n")
                if len(listFN) != 0:
                    f.write("########### FN\n")
                for l in listFN:
                    f.write(l)







errorAnalysisForRelation("adverse", '/Users/feili/Desktop/umass/MADE/made_test_data', 'output_other_adv/results.pkl', 'pretrain_other', './error_analysis/cancer_adverse')
errorAnalysisForRelation("reason", '/Users/feili/Desktop/umass/MADE/made_test_data', 'output_other_adv/results.pkl', 'pretrain_other', './error_analysis/cancer_reason')

# errorAnalysisForRelation("adverse", '/Users/feili/Desktop/umass/bioC_data/Cardio_test', 'output_other_made/results.pkl', 'pretrain_other_made', './error_analysis/cardio_adverse')
# errorAnalysisForRelation("reason", '/Users/feili/Desktop/umass/bioC_data/Cardio_test', 'output_other_made/results.pkl', 'pretrain_other_made', './error_analysis/cardio_reason')