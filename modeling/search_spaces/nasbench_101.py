"""
NASBench-101 search space, rollout, controller, evaluator.
During the development,
referred https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
"""

import os
import re
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nasbench import api
from nasbench.lib import graph_util, config
from .common import BaseRollout, SearchSpace


VERTICES = 7
MAX_EDGES = 9
_nasbench_cfg = config.build_config()


def _literal_np_array(arr):
    if arr is None:
        return None
    return "np.array({})".format(np.array2string(arr, separator=",").replace("\n", " "))


class _ModelSpec(api.ModelSpec):
    def __repr__(self):
        return "_ModelSpec({}, {}; pruned_matrix={}, pruned_ops={})".format(
            _literal_np_array(self.original_matrix), self.original_ops,
            _literal_np_array(self.matrix), self.ops)

    def hash_spec(self, *args, **kwargs):
        return super(_ModelSpec, self).hash_spec(_nasbench_cfg["available_ops"])


class NasBench101SearchSpace(SearchSpace):

    def __init__(self, base_dir, load_nasbench=True, multi_fidelity=False,
                 compare_reduced=True, compare_use_hash=False, validate_spec=True):
        super(NasBench101SearchSpace, self).__init__()

        self.ops_choices = [
            "conv1x1-bn-relu",
            "conv3x3-bn-relu",
            "maxpool3x3",
            "none",
        ]
        # operations: "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.base_dir = base_dir
        self.load_nasbench = load_nasbench
        self.multi_fidelity = multi_fidelity
        self.compare_reduced = compare_reduced
        self.compare_use_hash = compare_use_hash
        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 3 + 1 (none)
        self.num_ops = self.num_vertices - 2  # 5
        self.idx = np.triu_indices(self.num_vertices, k=1)
        self.validate_spec = validate_spec

        if self.load_nasbench:
            self._init_nasbench()

    def pad_archs(self, archs):
        return [self._pad_arch(arch) for arch in archs]

    def _pad_arch(self, arch):
        # padding for batchify training
        adj, ops = arch
        # all normalize the the reduced one
        spec = self.construct_modelspec(edges=None, matrix=adj, ops=ops)
        adj, ops = spec.matrix, self.op_to_idx(spec.ops)
        num_v = adj.shape[0]
        if num_v < VERTICES:
            padded_adj = np.concatenate((adj[:-1],
                                         np.zeros((VERTICES - num_v, num_v), dtype=np.int8),
                                         adj[-1:]))
            padded_adj = np.concatenate((padded_adj[:, :-1],
                                         np.zeros((VERTICES, VERTICES - num_v)),
                                         padded_adj[:, -1:]), axis=1)
            padded_ops = ops + [3] * (7 - num_v)
            adj, ops = padded_adj, padded_ops
        return adj, ops

    def _random_sample_ori(self):
        while 1:
            matrix = np.random.choice([0, 1], size=(self.num_vertices, self.num_vertices))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(self.ops_choices[:-1], size=(self.num_vertices)).tolist()
            ops[0] = "input"
            ops[-1] = "output"
            spec = _ModelSpec(matrix=matrix, ops=ops)
            if self.validate_spec and not self.nasbench.is_valid(spec):
                continue
            return NasBench101Rollout(
                spec.original_matrix, ops=self.op_to_idx(spec.original_ops), search_space=self)

    def _random_sample_me(self):
        while 1:
            splits = np.array(
                sorted([0] + list(np.random.randint(
                    0, self.max_edges + 1,
                    size=self.num_possible_edges - 1)) + [self.max_edges]))
            edges = np.minimum(splits[1:] - splits[:-1], 1)
            matrix = self.edges_to_matrix(edges)
            ops = np.random.randint(0, self.num_op_choices, size=self.num_ops)
            rollout = NasBench101Rollout(
                matrix, ops, search_space=self)
            try:
                self.nasbench._check_spec(rollout.genotype)
            except api.OutOfDomainError:
                # ignore out-of-domain archs (disconnected)
                continue
            else:
                return rollout

    # optional API
    def genotype_from_str(self, genotype_str):
        return eval(re.search("(_ModelSpec\(.+);", genotype_str).group(1) + ")")

    # ---- APIs ----
    def random_sample(self):
        return self._random_sample_ori()

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        matrix, ops = arch
        return self.construct_modelspec(edges=None, matrix=matrix, ops=ops)

    def rollout_from_genotype(self, genotype):
        return NasBench101Rollout(genotype.original_matrix,
                                  ops=self.op_to_idx(genotype.original_ops),
                                  search_space=self)

    def plot_arch(self, genotypes, filename, label, plot_format="pdf", **kwargs):
        graph = genotypes.visualize()
        graph.format = "pdf"
        graph.render(filename, view=False)
        return filename + ".{}".format(plot_format)

    def distance(self, arch1, arch2):
        pass

    # ---- helpers ----
    def _init_nasbench(self):
        # the arch -> performances dataset
        if self.multi_fidelity:
            self.nasbench = api.NASBench(os.path.join(self.base_dir, "nasbench_full.tfrecord"))
        else:
            self.nasbench = api.NASBench(os.path.join(self.base_dir, "nasbench_only108.tfrecord"))

    def edges_to_matrix(self, edges):
        matrix = np.zeros([self.num_vertices, self.num_vertices], dtype=np.int8)
        matrix[self.idx] = edges
        return matrix

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops if op not in {"input", "output"}]

    def matrix_to_edges(self, matrix):
        return matrix[self.idx]

    def construct_modelspec(self, edges, matrix, ops):
        if matrix is None:
            assert edges is not None
            matrix = self.edges_to_matrix(edges)

        # expect(graph_util.num_edges(matrix) <= self.max_edges,
        #        "number of edges could not exceed {}".format(self.max_edges))

        labeling = [self.ops_choices[op_ind] for op_ind in ops]
        labeling = ["input"] + list(labeling) + ["output"]
        model_spec = _ModelSpec(matrix, labeling)
        return model_spec

    def random_sample_arch(self):
        # not uniform, and could be illegal,
        #   if there is not edge from the INPUT or no edge to the OUTPUT,
        # Just check and reject for now
        return self.random_sample().arch

    def batch_rollouts(self, batch_size, shuffle=True, max_num=None):
        len_ = ori_len_ = len(self.nasbench.fixed_statistics)
        if max_num is not None:
            len_ = min(max_num, len_)
        list_ = list(self.nasbench.fixed_statistics.values())
        indexes = np.arange(ori_len_)
        np.random.shuffle(indexes)
        ind = 0
        while ind < len_:
            end_ind = min(len_, ind + batch_size)
            yield [NasBench101Rollout(
                list_[r_ind]["module_adjacency"],
                self.op_to_idx(list_[r_ind]["module_operations"]),
                search_space=self)
                for r_ind in indexes[ind:end_ind]]
            ind = end_ind

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]


class NasBench101Rollout(BaseRollout):

    def __init__(self, matrix, ops, search_space):
        super(NasBench101Rollout, self).__init__()
        
        self.arch = (matrix, ops)
        self.search_space = search_space
        self.perf = collections.OrderedDict()
        self._genotype = None

    def set_candidate_net(self, c_net):
        raise Exception("Should not be called")

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(
            self.genotype, filename,
            label=label, edge_labels=edge_labels)

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    def __repr__(self):
        return "NasBench101Rollout(matrix={arch}, perf={perf})" \
            .format(arch=self.arch, perf=self.perf)

    def __eq__(self, other):
        if self.search_space.compare_reduced:
            if self.search_space.compare_use_hash:
                # compare using hash, isomorphic archs would be equal
                return self.genotype.hash_spec() == other.genotype.hash_spec()
            else:
                # compared using reduced archs
                return (np.array(self.genotype.matrix).tolist() == np.array(other.genotype.matrix).tolist()) \
                       and list(self.genotype.ops) == list(other.genotype.ops)

        # compared using original/non-reduced archs, might be wrong
        return (np.array(other.arch[0]).tolist(), list(other.arch[1])) == \
               (np.array(self.arch[0]).tolist(), list(self.arch[1]))

