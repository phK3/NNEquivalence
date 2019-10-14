import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from k_means_scipy_distances import kmeanssample


class ClusterTree:
    def __init__(self, center):
        self.center = center
        self.children = []
        self.size = 1
        self.elements = -1
        self.avg_dist = -1

    def set_num_elements(self, elements):
        self.elements = elements

    def set_avg_dist(self, avg_dist):
        self.avg_dist = avg_dist

    def density(self):
        return self.elements / self.avg_dist

    def add_children(self, centers):
        self.children += [ClusterTree(c) for c in centers]
        self.size += len(centers)

    def add_child_clusters(self, clusters):
        self.children += clusters
        for c in clusters:
            self.size += c.size

    def get_children(self):
        return self.children

    def get_leaves(self):
        if not self.children:
            return self
        else:
            return [c.getLeaves() for c in self.children]

    def print_cluster_tree(self, depth):
        if not self.children:
            return '-' * depth + 'center: {center}'.format(center=str(self.center))
        else:
            ch_string = '\n'.join([child.print_cluster_tree(depth + 1) for child in self.children])
            return '-' * depth + 'center: {center}\n{ch}'.format(center=str(self.center), ch=ch_string)

    def __repr__(self):
        return self.print_cluster_tree(0)


class KMeansClassifier(ABC):

    def __init__(self, nclusters):
        self.nclusters = nclusters
        self.centers = []
        self.distances = []
        self.xtoc = []

    def set_num_clusters(self, nclusters):
        self.nclusters = nclusters

    @abstractmethod
    def fit_predict(self, data):
        pass

    @abstractmethod
    def get_distances(self):
        pass

    @abstractmethod
    def get_centers(self):
        pass


class ScikitKMeans(KMeansClassifier):

    def __init__(self, nclusters):
        super(ScikitKMeans, self).__init__(nclusters)
        self.classifier = KMeans(n_clusters=self.nclusters)
        self.data = None

    def fit_predict(self, data):
        self.data = data
        self.xtoc = self.classifier.fit_predict(self.data)
        self.centers = self.classifier.cluster_centers_

        return self.xtoc

    def get_distances(self):
        # don't calculate distances in fit_predict(), as maybe we don't need them
        # and it is faster without computing them
        if not self.distances:
            self.distances = min([np.linalg.norm(c - d) for d in self.data for c in self.centers])

        return self.distances

    def get_centers(self):
        return self.centers


class CustomKMeans(KMeansClassifier):

    def __init__(self, nclusters, kmsample=100, kmdelta=1e-4, kmiter=20, metric='euclidean', verbose=0):
        super(CustomKMeans, self).__init__(nclusters)
        self.kmsample = kmsample  # 0: random centres, > 0: kmeanssample
        self.kmdelta = kmdelta
        self.kmiter = kmiter
        self.metric = metric  # "cityblock" = manhattan, "chebyshev" = max, "cityblock" = L1,  Lqmetric
        self.verbose = verbose

    def fit_predict(self, data):
        centers, xtoc, dist = kmeanssample(data.to_numpy(), self.nclusters, nsample=self.kmsample,
                                           delta=self.kmdelta, maxiter=self.kmiter, metric=self.metric,
                                           verbose=self.verbose)

        self.centers = centers
        self.distances = dist
        self.xtoc = xtoc

        return xtoc

    def get_distances(self):
        # distances in this algorithm easily obtainable therefore direct computation in fit_predict()
        return self.distances

    def get_centers(self):
        return self.centers


class RecursiveClustering:

    def __init__(self, classifier):
        self.classifier = classifier

    def recursive_cluster(self, data, labels, purity, verbose=False, depth=0):
        num_labels = np.unique(labels.values).size
        self.classifier.set_num_clusters(num_labels)

        clusters = self.classifier.fit_predict(data)
        centers = self.classifier.get_centers()
        distances = self.classifier.get_distances()

        ct = [ClusterTree(c) for c in centers]

        for i in range(num_labels):
            mask = (clusters == i)
            elements, counts = np.unique(labels[mask].values, return_counts=True)
            el_counts = sorted(zip(elements, counts), key=lambda x: -x[1])
            current_purity = el_counts[0][1] / np.sum(counts)

            ct[i].set_num_elements(np.sum(counts))
            ct[i].set_avg_dist(np.average(distances[mask]))

            if verbose:
                print('-' * depth + 'mode = {m}, purity = {p}, size = {s}, avg_dist = {avg}'.format(m=el_counts[0][0],
                                                p=current_purity, s=np.sum(counts), avg=np.average(distances[mask])))

            if current_purity < purity:
                children = self.recursive_cluster(data[mask], labels[mask], purity, verbose=verbose, depth=depth + 1)
                ct[i].add_child_clusters(children)

        return ct