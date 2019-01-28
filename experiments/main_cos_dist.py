import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.metrics import calc_eer, cos_similarity, euclid_similarity
from utils.data_processing import load_data


if __name__ == '__main__':
    data, labels = load_data('../data', 'test')

    imposters = []
    targets = []

    all_files = data.shape[0]

    for i in tqdm(range(all_files)):
        current_label = labels[i]
        for j in range(all_files):
            if i == j:
                continue
            score = cos_similarity(data[i], data[j])
            if current_label == labels[j]:
                targets.append(score)
            else:
                imposters.append(score)

    err, fa, fr, thrs = calc_eer(targets, imposters)

    plt.plot(fa, thrs, color='r')
    plt.plot(fr, thrs, color='b')

    plt.ylabel('threshold')
    plt.xlabel('error')

    plt.title('EER = {}%'.format(err*100))
    plt.show()

