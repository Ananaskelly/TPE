import numpy as np


class TPEBatcher:

    def __init__(self, path_to_proto, data_root, batch_size=28):

        self.proto_path = path_to_proto
        self.data_root = data_root

        self.valid_part = 0.1

        self.train_list = []
        self.valid_list = []

        self.split()

        self.len_train_set = len(self.train_list)
        self.len_valid_set = len(self.valid_list)

        self.generator = self.__iter__()

    def split(self):

        with open(self.proto_path, 'r') as proto_file:

            all_lines = proto_file.readlines()
            np.random.shuffle(all_lines)

            num_ex = len(all_lines)
            bound = int(num_ex*(1 - self.valid_part))

            self.train_list = [line.split(',') for line in all_lines[:bound]]
            self.valid_list = [line.split(',') for line in all_lines[bound:]]

    def next_batch(self):

        return next(self.generator)

    def __iter__(self):

        idx = 0

        if idx < self.len_train_set:

            yield (np.load(self.train_list[idx][0]), np.load(self.train_list[idx][1]), np.load(self.train_list[idx][2]))

        else:
            idx = 0
