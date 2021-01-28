from __future__ import print_function, absolute_import

import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class NasBench101Dataset(Dataset):
    def __init__(self, root, train=True, full_vertex=False, valid_ratio=0.1, train_ratio=1.0,
                 search_space=None, minus=None, div=None):
        super(NasBench101Dataset, self).__init__()

        self.root = root
        self.train = train
        self.full_vertex = full_vertex
        self.valid_ratio = valid_ratio
        self.train_ratio = train_ratio
        self.search_space = search_space
        self.minus = minus
        self.div = div

        if self.full_vertex:
            self.train_file = os.path.join(root, 'nasbench_allv_train_v{:.2f}.pkl'.format(valid_ratio))
            self.valid_file = os.path.join(root, 'nasbench_allv_valid_v{:.2f}.pkl'.format(valid_ratio))
        else:
            self.train_file = os.path.join(root, 'nasbench_7v_train_v{:.2f}.pkl'.format(valid_ratio))
            self.valid_file = os.path.join(root, 'nasbench_7v_valid_v{:.2f}.pkl'.format(valid_ratio))

        if not (os.path.exists(self.train_file) and os.path.exists(self.valid_file)):
            print("Preparing pkl cache files {:s} and {:s} ...".format(self.train_file, self.valid_file))
            self._prepare()
            print("Done.")

        if self.train:
            self.cache_file = self.train_file
        else:
            self.cache_file = self.valid_file

        with open(self.cache_file, 'rb') as rf:
            print("Loading pkl cache from {:s} ...".format(self.cache_file))
            self.data = pickle.load(rf)
            print("Done.")
            if self.train and self.train_ratio is not None:
                self.data = self.data[:int(len(self.data) * self.train_ratio)]
            print("Number of arch data: {:d}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.minus is not None:
            data = (data[0], data[1] - self.minus, data[2] - self.minus)
        if self.div is not None:
            data = (data[0], data[1] / self.div, data[2] / self.div)
        return data

    def _prepare(self):
        fixed_statistics = list(self.search_space.nasbench.fixed_statistics.items())
        if not self.full_vertex:
            # only handle archs with 7 nodes for efficient batching
            fixed_statistics = [stat for stat in fixed_statistics if stat[1]["module_adjacency"].shape[0] == 7]

        print("Number of arch data: {:d}".format(len(fixed_statistics)))
        num_valid = int(len(fixed_statistics) * self.valid_ratio)
        num_train = len(fixed_statistics) - num_valid
        print("Number of train data: {:d}".format(num_train))
        print("Number of valid data: {:d}".format(num_valid))

        train_data = []
        for key, f_metric in fixed_statistics[:num_train]:
            num_v = f_metric["module_adjacency"].shape[0]
            if num_v < 7:
                padded_adj = np.concatenate((f_metric["module_adjacency"][:-1],
                                             np.zeros((7 - num_v, num_v), dtype=np.int8),
                                             f_metric["module_adjacency"][-1:]))
                padded_adj = np.concatenate((padded_adj[:, :-1], np.zeros((7, 7 - num_v)), padded_adj[:, -1:]), axis=1)
                padded_ops = f_metric["module_operations"][:-1] + ["none"] * (7 - num_v) + f_metric["module_operations"][-1:]
            else:
                padded_adj = f_metric["module_adjacency"]
                padded_ops = f_metric["module_operations"]
            arch = (padded_adj, self.search_space.op_to_idx(padded_ops))
            metrics = self.search_space.nasbench.computed_statistics[key]
            valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
            half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"] for i in range(3)])
            train_data.append((arch, valid_acc, half_valid_acc))

        valid_data = []
        for key, f_metric in fixed_statistics[num_train:]:
            num_v = f_metric["module_adjacency"].shape[0]
            if num_v < 7:
                padded_adj = np.concatenate((f_metric["module_adjacency"][:-1],
                                             np.zeros((7 - num_v, num_v), dtype=np.int8),
                                             f_metric["module_adjacency"][-1:]))
                padded_adj = np.concatenate((padded_adj[:, :-1], np.zeros((7, 7 - num_v)), padded_adj[:, -1:]), axis=1)
                padded_ops = f_metric["module_operations"][:-1] + ["none"] * (7 - num_v) + f_metric["module_operations"][-1:]
            else:
                padded_adj = f_metric["module_adjacency"]
                padded_ops = f_metric["module_operations"]
            arch = (padded_adj, self.search_space.op_to_idx(padded_ops))
            metrics = self.search_space.nasbench.computed_statistics[key]
            valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
            half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"] for i in range(3)])
            valid_data.append((arch, valid_acc, half_valid_acc))

        with open(self.train_file, 'wb') as wf:
            pickle.dump(train_data, wf)
        with open(self.valid_file, 'wb') as wf:
            pickle.dump(valid_data, wf)

