import numpy as np


class TPEBatcher:

    def __init__(self, path_to_proto, data_root, batch_size=28):

        self.proto_path = path_to_proto
        self.data_root = data_root
        self.global_step = 0

        with open(path_to_proto, 'r') as proto_file:
            lines = proto_file.readlines()

            self.num_ex = len(lines)

        self.generator = self.__iter__()

    def next_batch(self):

        return next(self.generator)

    def __iter__(self):

        if self.global_step < self.num_ex:
            pass
        else:
            self.global_step = 0
