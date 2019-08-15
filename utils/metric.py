import random

import keras.backend as K
import math

shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_dummy(size=14, num_fixations=100, num_salience_points=200):
    # first generate dummy gt and salience map
    discrete_gt = np.zeros((size, size))
    s_map = np.zeros((size, size))

    for i in range(0, num_fixations):
        discrete_gt[np.random.randint(size), np.random.randint(size)] = 1.0

    for i in range(0, num_salience_points):
        s_map[np.random.randint(size), np.random.randint(size)] = 255 * round(random.random(), 1)

    # check if gt and s_map are same size
    assert discrete_gt.shape == s_map.shape, 'sizes of ground truth and salience map don\'t match'
    return s_map, discrete_gt


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
    return norm_s_map


def discretize_gt(gt):
    import warnings
    warnings.warn('can improve the way GT is discretized')
    return gt / 255


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    # gt = discretize_gt(gt)
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        assert np.max(gt) <= 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(s_map) <= 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
    # tp_list.append(tp)
    # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_borji(s_map, gt, splits=100, stepsize=0.1):
    # gt = discretize_gt(gt)
    num_fixations = np.sum(gt)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        for k in range(0, int(num_fixations)):
            temp_list.append(np.random.randint(num_pixels))
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[int(k % s_map.shape[0] - 1), int(k / s_map.shape[0])])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)

def auc_shuff(s_map, gt, shufMap, stepSize=.01):
    """
    Computer SAUC score. A simple implementation
    :param gt : list of fixation annotataions
    :param s_map : list only contains one element: the result annotation - predicted saliency map
    :return score: int : score
    """
    # resAnn = resAnn / 254
    salMap = s_map - np.min(s_map)
    if np.max(salMap) > 0:
        salMap = salMap / (np.max(salMap) - np.min(salMap))
    xd, yd = np.where(gt > 0)
    Sth = np.asarray([salMap[x][y] for x, y in zip(xd, yd)])
    Nfixations = len(gt)

    others = np.copy(shufMap)
    for x, y in zip(xd, yd):
        others[x][y] = 0

    ind = np.nonzero(others)  # find fixation locations on other images
    nFix = shufMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes) + 2)
    fp = np.zeros(len(allthreshes) + 2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(nFix[randfix >= thresh])) / Nothers for thresh in allthreshes]

    auc = np.trapz(tp, fp)
    return auc


def nss_metric(gt, s_map):
    # gt = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)

    x, y = np.where(gt == 1)
    temp = []
    for i in zip(x, y):
        temp.append(s_map_norm[i[0], i[1]])
    return np.mean(temp)


def infogain(s_map, gt, baseline_map):
    # gt = discretize_gt(gt)
    # assuming s_map and baseline_map are normalized
    eps = 2.2204e-16

    s_map = s_map / (np.sum(s_map) * 1.0)
    baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

    # for all places where gt=1, calculate info gain
    temp = []
    x, y = np.where(gt == 1)
    for i in zip(x, y):
        temp.append(np.log2(eps + s_map[i[0], i[1]]) - np.log2(eps + baseline_map[i[0], i[1]]))

    return np.mean(temp)


def similarity(s_map, gt):
    # here gt is not discretized nor normalized
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    x, y = np.where(gt > 0)
    sim = 0.0
    for i in zip(x, y):
        sim = sim + min(gt[i[0], i[1]], s_map[i[0], i[1]])
    return sim


def cc(gt, s_map):
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    a = s_map_norm
    b = gt_norm
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum());
    return r


# def kldiv(gt, s_map):
#     s_map = s_map/np.max(s_map)
#     s_map = s_map / (np.sum(s_map) * 1.0)
#     gt = gt / (np.sum(gt) * 1.0)
#     eps = 2.2204e-16
#     return np.sum(gt * np.log(eps + gt / (s_map + eps)))

def kldiv(gt, s_map):
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    eps = 2.2204e-16
    return np.sum(gt * np.log(eps + gt / (s_map + eps)))


def kldivn(y_true, y_pred):
    with tf.Session() as sess:
        y_pred = np.float64(y_pred)
        max_y_pred = K.repeat_elements(
            K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=0), axis=0)),
                                            shape_r_out, axis=-1)), shape_c_out, axis=-1)
        y_pred /= max_y_pred

        sum_y_true = K.repeat_elements(
            K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=0), axis=0)),
                                            shape_r_out, axis=-1)), shape_c_out, axis=-1)
        sum_y_pred = K.repeat_elements(
            K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=0), axis=0)),
                                            shape_r_out, axis=-1)), shape_c_out, axis=-1)
        y_true /= (sum_y_true + K.epsilon())
        y_pred /= (sum_y_pred + K.epsilon())

        return K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1).eval()


def calcsauc_score(gtsAnn, resAnn, shufMap, stepSize=.01):
    """
    Computer SAUC score. A simple implementation
    :param gtsAnn : list of fixation annotataions
    :param resAnn : list only contains one element: the result annotation - predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)
    Sth = np.asarray([salMap[y - 1][x - 1] for y, x in gtsAnn])
    Nfixations = len(gtsAnn)

    others = np.copy(shufMap)
    for y, x in gtsAnn:
        others[y - 1][x - 1] = 0

    ind = np.nonzero(others)  # find fixation locations on other images
    nFix = shufMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes) + 2)
    fp = np.zeros(len(allthreshes) + 2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(nFix[randfix >= thresh])) / Nothers for thresh in allthreshes]

    auc = np.trapz(tp, fp)
    return auc