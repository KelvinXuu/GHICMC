import math
import os

import numpy as np
from numpy.random import randint
import torch
import logging
import random


def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()


def setup_seed(seed_n):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def write_eva(filepath, ms, *arg):

    with open(filepath, 'a') as file:

        file.write('missing-rate:{:.1f} \t '.format(ms))
        output = 'ACC_mean:'+str(round(np.mean(arg[0]) * 100, 2)) + ',  ' + 'ACC_std:'+str(round(np.std(arg[0]) * 100, 2)) + ';  ' + \
                'NMI_mean:'+str(round(np.mean(arg[1]) * 100, 2)) + ',  ' + 'NMI_std:'+str(round(np.std(arg[1]) * 100, 2)) + ';  ' + \
                'ARI_mean:'+str(round(np.mean(arg[2]) * 100, 2)) + ',  ' + 'ARI_std:'+str(round(np.std(arg[2]) * 100, 2)) + ';\n'
        file.write(output)
        file.flush()


def get_mask( data_size, missing_ratio,view_num):
    """
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size*missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask


def get_logger(config, main_dir='../logs/'):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        main_dir + str(config['dataset']) + '_' + str(config['missing_rate']).replace('.', '_') + '.logs')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 3:
        logger.info("ACC"+str(arg[0]))
        logger.info("NMI"+str(arg[1]))
        logger.info("ARI"+str(arg[2]))
        output = """ 
                     ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f} 
                     ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100, np.mean(arg[1]) * 100,
                                                     np.std(arg[1]) * 100, np.mean(arg[2]) * 100, np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';\n'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)