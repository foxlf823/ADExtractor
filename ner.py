import preprocess
import os
import bioc
import shutil
from utils.data import Data
from utils.functions import normalize_word
from options import opt
import torch
from model.seqmodel import SeqModel
import numpy as np
import torch.autograd as autograd
import logging
import random
import gc
import torch.nn as nn
import torch.optim as optim
import time
import sys
from utils.metric import get_ner_fmeasure

def featureCapital(word):
    if word[0].isalpha() and word[0].isupper():
        return 1
    else:
        return 0

ENTITY_TYPE = set(['Severity', 'Route', 'Drug', 'Dose', 'Frequency', 'Indication', 'Duration', 'ADE', 'SSLIF'])
def getLabel(start, end, sent_entity):
    """
    Only considering the entity in ENTITY_TYPE. For double annotation, the first-meet entity is considered.
    :param start:
    :param end:
    :param sent_entity:
    :return:
    """
    match = ""
    for index, entity in sent_entity.iterrows():
        if start == entity['start'] and end == entity['end'] : # S
            match = "S"
            break
        elif start == entity['start'] and end != entity['end'] : # B
            match = "B"
            break
        elif start != entity['start'] and end == entity['end'] : # E
            match = "E"
            break
        elif start > entity['start'] and end < entity['end']:  # M
            match = "M"
            break

    if match != "" and sent_entity.loc[index]['type'] in ENTITY_TYPE:
        return match+"-"+sent_entity.loc[index]['type']
    else:
        return "O"

def generateData(tokens, entitys, names, output_file):
    """
    generate data for NCRFpp
    :param token: df token
    :param entity: df entity
    :param name: df name
    :param output_file: where to save the result
    :return:
    """
    f = open(output_file, 'w')

    for i, name in enumerate(names):
        doc_token = tokens[i]
        doc_entity = entitys[i]

        for sent_idx in range(9999): # this is an assumption, may be failed
            sent_token = doc_token[(doc_token['sent_idx'] == sent_idx)]
            sent_entity = doc_entity[(doc_entity['sent_idx'] == sent_idx)]

            if sent_token.shape[0] == 0:
                break

            for _, token in sent_token.iterrows():
                word = token['text']
                pos = token['postag']
                cap = featureCapital(word)
                label = getLabel(token['start'], token['end'], sent_entity)

                f.write("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))

            f.write("\n")

    f.close()

def generateDataForOneDoc(doc_token, doc_entity):

    lines = []

    for sent_idx in range(9999): # this is an assumption, may be failed
        sent_token = doc_token[(doc_token['sent_idx'] == sent_idx)]
        sent_entity = doc_entity[(doc_entity['sent_idx'] == sent_idx)]

        if sent_token.shape[0] == 0:
            break

        for _, token in sent_token.iterrows():
            word = token['text']
            pos = token['postag']
            cap = featureCapital(word)
            label = getLabel(token['start'], token['end'], sent_entity)

            lines.append("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))

        lines.append("\n")
    return lines


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    with torch.no_grad():
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        features = [np.asarray(sent[1]) for sent in input_batch_list]
        feature_num = len(features[0][0])
        chars = [sent[2] for sent in input_batch_list]
        labels = [sent[3] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(map(len, words))
        max_seq_len = word_seq_lengths.max()
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        feature_seq_tensors = []
        for idx in range(feature_num):
            feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len))).long())
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        label_seq_tensor = label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]
        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len.item()-len(chars[idx])) for idx in range(len(chars))]
        length_list = [map(len, pad_char) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len.item(),-1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len.item(),)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        if gpu:
            word_seq_tensor = word_seq_tensor.cuda()
            for idx in range(feature_num):
                feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
            word_seq_lengths = word_seq_lengths.cuda()
            word_seq_recover = word_seq_recover.cuda()
            label_seq_tensor = label_seq_tensor.cuda()
            char_seq_tensor = char_seq_tensor.cuda()
            char_seq_recover = char_seq_recover.cuda()
            mask = mask.cuda()
        return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print " Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print "tag:",tag_seq
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores

def train(data):
    print "Training model..."
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)
    model = SeqModel(data)
    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(0)
    best_dev = -10
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char,
                                                          batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data.item()
            total_loss += loss.data.item()
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
                    exit(0)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
        end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
        idx, epoch_cost, train_num / epoch_cost, total_loss))
        print "totalloss:", total_loss
        if total_loss > 1e8 or str(total_loss) == "nan":
            print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
            exit(0)
        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev

            model_name = data.model_dir + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        gc.collect()

def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    # print pred_variable.size()
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label

def evaluateWhenTest(data, model):

    instances = data.raw_Ids
    nbest_pred_results = []
    model.eval()
    batch_size = data.HP_batch_size

    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)

        scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, data.nbest)
        nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
        nbest_pred_results += nbest_pred_result

    return nbest_pred_results

def read_instanceFromBuffer(in_lines, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    feature_num = len(feature_alphabets)
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            ## get features
            feat_list = []
            feat_Id = []
            for idx in range(feature_num):
                feat_idx = pairs[idx+1].split(']',1)[-1]
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
            features.append(feat_list)
            feature_Ids.append(feat_Id)
            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, features, chars, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def checkWrongState(labelSequence):
    positionNew = -1
    positionOther = -1
    currentLabel = labelSequence[-1]
    assert currentLabel[0] == 'M' or currentLabel[0] == 'E'

    for j in range(len(labelSequence)-1)[::-1]:
        if positionNew == -1 and currentLabel[2:] == labelSequence[j][2:] and labelSequence[j][0] == 'B' :
            positionNew = j
        elif positionOther == -1 and (currentLabel[2:] != labelSequence[j][2:] or labelSequence[j][0] != 'M'):
            positionOther = j

        if positionOther != -1 and positionNew != -1:
            break

    if positionNew == -1:
        return False
    elif positionOther < positionNew:
        return True
    else:
        return False

def translateNCRFPPintoBioc(doc_token, predict_results, file_name):
    collection = bioc.BioCCollection()
    document = bioc.BioCDocument()
    collection.add_document(document)
    document.id = file_name
    passage = bioc.BioCPassage()
    document.add_passage(passage)
    passage.offset = 0
    entity_id = 1

    sent_num = len(predict_results)
    for idx in range(sent_num):
        sent_length = len(predict_results[idx][0])
        sent_token = doc_token[(doc_token['sent_idx'] == idx)]
        assert sent_token.shape[0] == sent_length
        labelSequence = []

        for idy in range(sent_length):
            token = sent_token.iloc[idy]
            label = predict_results[idx][0][idy]
            labelSequence.append(label)

            if label[0] == 'S' or label[0] == 'B':
                anno_entity = bioc.BioCAnnotation()
                passage.add_annotation(anno_entity)
                anno_entity.id = str(entity_id)
                anno_entity.infons['type'] = label[2:]
                anno_entity_location = bioc.BioCLocation(token['start'], token['end']-token['start'])
                anno_entity.add_location(anno_entity_location)
                anno_entity.text = token['text']
                entity_id += 1

            elif label[0] == 'M' or label[0] == 'E':
                if checkWrongState(labelSequence):
                    anno_entity = passage.annotations[-1]

                    whitespacetoAdd = token['start'] - anno_entity.locations[0].end
                    for _ in range(whitespacetoAdd):
                        anno_entity.text += " "
                    anno_entity.text += token['text']
                    anno_entity.locations[0].length = token['end'] - anno_entity.locations[0].offset



    bioc_file = open(file_name + ".bioc.xml", 'w')
    bioc.dump(collection, bioc_file)
    bioc_file.close()


logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

logging.info(opt)

if opt.whattodo==1:
    # step 1, translate the MADE data into NCRFpp data, both train and test
    train_token, train_entity, _, train_name = preprocess.loadPreprocessData(opt.traindata)
    train_ner_file = os.path.join(opt.traindata, 'ncrfpp.txt')
    generateData(train_token, train_entity, train_name, train_ner_file)

    test_token, test_entity, _, test_name = preprocess.loadPreprocessData(opt.testdata)
    test_ner_file = os.path.join(opt.testdata, 'ncrfpp.txt')
    generateData(test_token, test_entity, test_name, test_ner_file)
elif opt.whattodo==2:

    # step 2, train a model via NCRFpp
    opt.config = "demo.train.config"
    data = Data()
    data.read_config(opt.config)
    status = data.status.lower()
    data.HP_gpu = torch.cuda.is_available()

    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.fix_alphabet()

    data.generate_instance('train')
    data.generate_instance('dev')
    data.build_pretrain_emb()
    train(data)


elif opt.whattodo==3:
    # step 3, evaluate on test data and output results in bioc format, one doc one file
    opt.config = "demo.decode.config"
    data = Data()
    data.read_config(opt.config)
    status = data.status.lower()
    data.HP_gpu = torch.cuda.is_available()
    data.load(data.dset_dir)
    data.read_config(opt.config)

    data.show_data_summary()
    data.fix_alphabet()
    model = SeqModel(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    ner_output_dir = os.path.join(opt.testdata, "ner")
    if os.path.exists(ner_output_dir):
        shutil.rmtree(ner_output_dir)
        os.makedirs(ner_output_dir)
    else:
        os.makedirs(ner_output_dir)

    test_token, test_entity, _, test_name = preprocess.loadPreprocessData(opt.testdata)
    for i, doc_name in enumerate(test_name):
        doc_token = test_token[i]
        doc_entity = test_entity[i]

        ncrf_data = generateDataForOneDoc(doc_token, doc_entity)

        data.raw_texts, data.raw_Ids = read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                     data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                     data.MAX_SENTENCE_LENGTH)

        decode_results = evaluateWhenTest(data, model)
        test_ner_file = os.path.join(ner_output_dir, doc_name)
        translateNCRFPPintoBioc(doc_token, decode_results, test_ner_file)




