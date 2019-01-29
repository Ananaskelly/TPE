import os
import glob
import csv
import itertools
import numpy as np


def txt_to_numpy(path_to_data, out_path):
    """
    Convert files to numpy

    :param path_to_data: path to directory with *.txt files
    :param out_path:     path to store *.npy files
    :return:
    """
    all_files = glob.glob(os.path.join(path_to_data, '*.txt'))

    for file in all_files:
        name = os.path.basename(file)[:-4]
        with open(file, 'r') as curr_file:
            line = curr_file.readlines()[0].strip().split(' ')
            arr = np.array(line[1:-1], dtype=np.float)
            np.save(os.path.join(out_path, name), arr)


def load_data(root, set_type):
    """

    :param root
    :param set_type:
    :return:
    """
    if set_type.lower() not in ['train', 'test']:
        raise Exception('Can\'t load data with such set type: {}'.format(set_type))

    path = os.path.join(root, '{}_db_numpy'.format(set_type))
    all_files = glob.glob(os.path.join(path, '*.npy'))
    data = []
    labels = []
    for file in all_files:
        name = os.path.basename(file)
        data.append(np.load(file))
        labels.append(int(name.split('_')[0].lstrip('0')))

    return np.array(data), np.array(labels)


def cross_arrays(all_ids, target_ids):

    ans = np.full(all_ids.shape, 1, dtype=np.int)
    for idx in target_ids:
        ans *= (all_ids != idx).astype(np.int)

    return np.squeeze(np.argwhere(ans))


def generate_tuples(path_to_files, path_to_out_proto_file):

    with open(path_to_out_proto_file, 'w'):
        pass

    all_file_names = os.listdir(path_to_files)
    ids = np.array([file.split('_')[0].lstrip('0') for file in all_file_names], dtype=np.int)
    unique_ids = np.uniquie(ids)
    num_unique_ids = unique_ids.shape

    for i in range(num_unique_ids):
        target_ids = np.squeeze(np.argwhere(ids == num_unique_ids[i]))
        target_pairs = itertools.combinations(target_ids, 2)

        imposter_ids = np.squeeze(np.argwhere(ids != num_unique_ids[i]))
        for pair in target_pairs:
            for idx in imposter_ids:
                with open(path_to_out_proto_file, 'a') as proto_file:
                    proto_file.write('{},{},{}\n'.format(all_file_names[pair[0]], all_file_names[pair[1]],
                                                         all_file_names[idx]))


if __name__ == '__main__':
    path_to_train_data = '../data/train_db'
    path_to_test_data = '../data/test_db'

    out_train = '../data/train_db_numpy'
    out_test = '../data/test_db_numpy'

    # txt_to_numpy(path_to_test_data, out_test)
    # txt_to_numpy(path_to_train_data, out_train)
    generate_tuples(path_to_train_data, '../data/train_tuples_proto.txt')
