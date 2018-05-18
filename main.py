
import preprocess
import logging
import random
import numpy as np
import torch
import os
import shutil


from options import opt

import trainandtest


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
    preprocess.preprocess(opt.traindata)
    preprocess.preprocess(opt.testdata)
elif opt.whattodo==2:
    if os.path.exists(opt.pretrain):
        shutil.rmtree(opt.pretrain)
        os.makedirs(opt.pretrain)
    else:
        os.makedirs(opt.pretrain)

    train_token, train_entity, train_relation, train_name = preprocess.loadPreprocessData(opt.traindata)
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(opt.testdata)

    trainandtest.pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation, test_name)
elif opt.whattodo==3:
    if os.path.exists(opt.output):
        shutil.rmtree(opt.output)
        os.makedirs(opt.output)
    else:
        os.makedirs(opt.output)


    trainandtest.train()
else:

    result_dumpdir = os.path.join(opt.testdata, "predicted")
    if os.path.exists(result_dumpdir):
            shutil.rmtree(result_dumpdir)
            os.makedirs(result_dumpdir)
    else:
        os.makedirs(result_dumpdir)

    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(opt.testdata)

    # trainandtest.test(test_token, test_entity, test_relation, test_name, result_dumpdir)
    #trainandtest.test1(test_token, test_entity, test_relation, test_name, result_dumpdir)
    trainandtest.test2(test_token, test_entity, test_relation, test_name, result_dumpdir)



