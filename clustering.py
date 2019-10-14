import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from scipy.spatial import distance
from expression_encoding import flatten
from k_means_scipy_distances import kmeanssample


class ClusterTree:
    def __init__(self, center):
        self.center = center
        self.children = []
        self.size = 1 # no of nodes in this clustertree
        self.elements = -1 # no of elements in cluster
        self.avg_dist = -1 # avg distance of elements to cluster center
        self.distance = -1 # distance of cluster to nearest other cluster

    def set_num_elements(self, elements):
        self.elements = elements

    def set_avg_dist(self, avg_dist):
        self.avg_dist = avg_dist

    def density(self):
        return self.elements / self.avg_dist

    def compute_cluster_distance(self, centers, metric='euclidean'):
        dists = distance.cdist(np.array([self.center]), np.array(centers), metric=metric).flatten()
        # return 2nd smallest element, smallest element should always be 0, as it is distance to self
        if len(dists) > 2:
            self.distance = np.partition(dists, 2)[1]
        else:
            self.distance = max(dists)

        for ch in self.children:
            ch.compute_cluster_distance([c.center for c in self.children], metric=metric)

        return self.distance

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
            return [c.get_leaves() for c in self.children]

    def print_cluster_tree(self, depth, verbose=False):
        node_string = 'center: {center}'.format(center=str(self.center))
        if verbose:
            node_string += ' density = {d}'.format(d=self.density())

        if not self.children:
            return '-' * depth + node_string
        else:
            ch_string = '\n'.join([child.print_cluster_tree(depth + 1, verbose) for child in self.children])
            return '-' * depth + node_string + '\n{ch}'.format(ch=ch_string)

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
        self.data = data.to_numpy()
        self.xtoc = self.classifier.fit_predict(self.data)
        self.centers = self.classifier.cluster_centers_

        return self.xtoc

    def get_distances(self):
        # don't calculate distances in fit_predict(), as maybe we don't need them
        # and it is faster without computing them
        self.distances = [min([np.linalg.norm(c - d) for c in self.centers]) for d in self.data]
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

    def __init__(self, classifier_type='SciKit', metric='euclidean'):
        self.classifier_type = classifier_type
        self.metric = metric

    def recursive_cluster(self, data, labels, purity, verbose=False, depth=0):
        num_labels = np.unique(labels.values).size

        if self.classifier_type == 'SciKit':
            classifier = ScikitKMeans(num_labels)
        elif self.classifier_type == 'Custom':
            if not self.metric in ['chebyshev', 'cityblock', 'euclidean']:
                raise ValueError('metric {m} is not supported by CustomKMeans'.format(m=self.metric))

            classifier = CustomKMeans(num_labels, len(data) // 40, metric=self.metric)
        else:
            classifier = None
        #classifier = self.classifier(num_labels)
        #self.classifier.set_num_clusters(num_labels)

        clusters = classifier.fit_predict(data)
        centers = classifier.get_centers()
        distances = np.array(classifier.get_distances())

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