
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


if opt.whattodo==1:
    preprocess.preprocess(opt.traindata)
    preprocess.preprocess(opt.testdata)
elif opt.whattodo==2:

    if os.path.exists(opt.output):
        shutil.rmtree(opt.output)
        os.makedirs(opt.output)
    else:
        os.makedirs(opt.output)

    train_token, train_entity, train_relation = preprocess.loadPreprocessData(opt.traindata)
    test_token, test_entity, test_relation = preprocess.loadPreprocessData(opt.testdata)

    trainandtest.train(opt, train_token, train_entity, train_relation, test_token, test_entity, test_relation)
else :
    test_token, test_entity, test_relation = preprocess.loadPreprocessData(opt.testdata)
    trainandtest.test(opt, test_token, test_entity, test_relation)





