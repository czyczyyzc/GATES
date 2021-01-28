import abc
import collections


class BaseRollout(metaclass=abc.ABCMeta):
    """
    Rollout is the interface object that is passed through all the components.
    """

    def __init__(self):
        self.perf = collections.OrderedDict()

    @abc.abstractmethod
    def set_candidate_net(self, c_net):
        pass

    @abc.abstractmethod
    def plot_arch(self, filename, label="", edge_labels=None):
        pass

    def get_perf(self, name="reward"):
        return self.perf.get(name, None)

    def set_perf(self, value, name="reward"):
        self.perf[name] = value
        return self

    def set_perfs(self, perfs):
        for n, v in perfs.items():
            self.set_perf(v, name=n)
        return self


class SearchSpace(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""

    @abc.abstractmethod
    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""

    @abc.abstractmethod
    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    @abc.abstractmethod
    def distance(self, arch1, arch2):
        pass

    @classmethod
    @abc.abstractmethod
    def supported_rollout_types(cls):
        pass

    def mutate(self, rollout, **mutate_kwargs):
        """
        Mutate a rollout to a neighbor rollout in the search space.
        Called by mutation-based controllers, e.g., EvoController.
        """
        # raise NotImplementedError()
        return

