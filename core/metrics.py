import numpy as np


def cos_similarity(a, b):
    """
    Calculate cosine similarity between input vectors

    :param a:
    :param b:
    :return: cosine similarity input vectors
    """
    if len(a.shape) != len(b.shape):
        raise Exception('Input shapes must be the same')
    if not np.equal(a.shape, b.shape):
        raise Exception('Input shapes must be the same')

    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def euclid_similarity(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    if len(a.shape) != len(b.shape):
        raise Exception('Input shapes must be the same')
    if not np.equal(a.shape, b.shape):
        raise Exception('Input shapes must be the same')

    return 1 / np.sqrt(np.sum((a-b)**2))


def calc_eer(targ_scores, imp_scores):
    min_score = np.minimum(np.min(targ_scores), np.min(imp_scores))
    max_score = np.maximum(np.max(targ_scores), np.max(imp_scores))

    n_targs = len(targ_scores)
    n_imps = len(imp_scores)

    num_points = 50
    fa = np.zeros((num_points,))
    fr = np.zeros((num_points,))

    thrs = np.linspace(min_score, max_score, num_points)

    min_gap = float('inf')
    eer = 0

    for i, thr in enumerate(thrs):
        cur_fa = len(np.where(imp_scores > thr)[0]) / n_imps
        cur_fr = len(np.where(targ_scores < thr)[0]) / n_targs
        fa[i] = cur_fa
        fr[i] = cur_fr
        gap = np.abs(cur_fa - cur_fr)
        if gap < min_gap:
            min_gap = gap
            eer = (cur_fa + cur_fr) / 2

    return eer, fa, fr, thrs
