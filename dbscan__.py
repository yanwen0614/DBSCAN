# -*- coding: utf-8 -*-


import logging
import multiprocessing
import warnings
from collections import Counter, deque

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from tqdm import tqdm

from multiprocess_df import Consumer, TaskTracker

logger = logging.getLogger(__name__)


class DBSCAN(ClusterMixin, BaseEstimator):
    """

    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.

    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    >>> clustering
    DBSCAN(eps=3, min_samples=2)
    """

    def __init__(self, eps=0.5, min_samples=15 ,min_dense=6 , metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,max_iter = 1e6,
                 n_jobs=None):
        # print("eps, min_samples ,min_dense:  ",eps, min_samples ,min_dense)
        self.eps = eps
        self.min_samples = min_samples
        self.min_dense = min_dense
        self.metric = metric
        self.max_iter = int(max_iter)
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """Perform DBSCAN clustering from features, or distance matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X, accept_sparse='csr')
        self.X = X

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            raise NotImplementedError

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        neighbors_model = NearestNeighbors(
            radius=self.eps, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric,
            metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X,
                                                         return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        # core_samples = np.asarray(n_neighbors >= self.min_samples,
        #                           dtype=np.uint8)
        core_samples, neighborhoods_core = self.check_core(n_neighbors,
                    neighborhoods, self.min_dense)
        self.dbscan_first(core_samples, neighborhoods_core, labels)

        logger.debug('dbscan_after')
        labels_ = tuple(np.full(X.shape[0], -1, dtype=np.intp))
        for i in range(self.max_iter):
            labels_ = labels.copy()
            labels = self.dbscan_after(core_samples, neighborhoods, labels)
            if False not in (labels_== labels):
                break

        logger.debug('{} iter dbscan_after'.format(str(i)))
        self.core_sample_indices_ = np.where(core_samples)[0]

        labels_counter = Counter(labels)
        cluster_labels = [ c_n[0] for c_n in  labels_counter.items() if c_n[1] > self.min_samples]
        labels = [ l if l in cluster_labels else -1 for l in labels ]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Perform DBSCAN clustering from features or distance matrix,
        and return cluster labels.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_

    def check_core(self,n_neighbors,neighbors,basic_dense):
        logger.debug("calc_neighbors_list")
        core = set()

        neighbors_ = tuple([] for i in n_neighbors)
        core_samples_index = np.where(n_neighbors >= basic_dense)[0]
        if len(core_samples_index)< 100000:
            for ind in core_samples_index:

                ind, neighbor = self.single_func(ind, neighbors=neighbors,basic_dense=basic_dense)
                neighbors_[ind].extend(neighbor)
                core.update(neighbor)
        else:
            logger.debug("begin multiprocess")
            args = {
                "neighbors": neighbors,
                "basic_dense": basic_dense
            }
            results_list = self.multiprocess_func(self.single_func, core_samples_index, **args)
            logger.debug("multiprocess finish")
            for results in results_list:
                ind, neighbor = results
                neighbors_[ind].extend(neighbor)
                core.update(neighbor)

        core_array = np.zeros(len(n_neighbors))
        for i in core:
            core_array[i] = 1
        logger.debug("get_neighbors_list")
        return core_array, neighbors_

    def single_func(self, ind,  neighbors=None, basic_dense=None):
        neighbors_counter = Counter(neighbors[ind])

        for ind_ in neighbors[ind]:
            neighbors_counter.update(neighbors[ind_])
        neighbor = tuple(ind_ for ind_, num in neighbors_counter.items() if num >= basic_dense)
        neighbors_set = set(neighbors[ind]).intersection(set(neighbor))

        coor = np.array([self.X[i] for i in neighbors_set])
        x_coor = coor[:,0]
        y_coor = coor[:,1]

        dx = max(x_coor)-min(x_coor)
        dy = max(y_coor)-min(y_coor)

        if dx>1000 or dy>1000:
            print(x)
            # from matplotlib import pyplot as plt
            # plt.plot(x_coor, y_coor, 'o',markersize=2)
            # plt.show()



        return ind, neighbors_set

    def multiprocess_func(self, single_func, listdata, **kwarg):

        num_process = self.n_jobs
        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()
        # Enqueue tasks
        num_task = len(listdata)
        for i in range(num_task):
            tasks.put(listdata[i])

        logger.debug('Create {} processes'.format(num_process))
        consumers = [Consumer(single_func, tasks, results, error_queue, **kwarg)
                    for i in range(num_process)]
        for w in consumers:
            w.start()
        # Add a task tracking process
        task_tracker = TaskTracker(tasks, False)
        task_tracker.start()
        # Wait for all input data to be processed
        tasks.join()
        for w in consumers:
            w.terminate()
        # If there is any error in any process, output the error messages
        num_error = error_queue.qsize()
        if num_error > 0:
            for i in range(num_error):
                logger.error(error_queue.get())
            raise RuntimeError('Multi process jobs failed')
        else:
            # Collect results
            result_table_file = open("result_table","w")
            logger.debug('Collect results')
            result_table = deque()
            for i in range(num_task):
                res = results.get()
                result_table.append(res)
            logger.debug('Collect results done')
            return result_table

    def dbscan_first(self, is_core, neighborhoods, labels,):
        """
        get cluster init
        """
        logger.debug("dbscan_first")
        i, label_num = 0, 0
        neighb = []
        stack = deque()

        for i in range(labels.shape[0]):
            if labels[i] != -1 or not is_core[i]:
                continue

            # Width-first search starting from i, ending at the non-core points.
            # This is very similar to the classic algorithm for computing connected
            # components, the difference being that we label non-core points as
            # part of a cluster (component), but don't expand their neighborhoods.
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:
                        neighb = neighborhoods[i]
                        for v in neighb:
                            if labels[v] == -1:
                                stack.append(v)

                if len(stack) == 0:
                    break
                i = stack.popleft()

            label_num += 1
        #     print("label_num",label_num,end="")
        #     print("\r")
        # print("\n")
        return labels

    def dbscan_after(self, is_core, neighborhoods, labels,):
        i, label_num = 0, 0
        neighb = []
        stack = deque()
        arrived = np.full(labels.shape[0], False, dtype=bool)
        arrived_num = 0

        for i in range(labels.shape[0]):
            if arrived[i]:
                 continue
            arrived[i] = True
            # Width-first search starting from i, ending at the non-core points.
            # This is very similar to the classic algorithm for computing connected
            # components, the difference being that we label non-core points as
            # part of a cluster (component), but don't expand their neighborhoods.
            # print("starte while")
            while_loop = 0
            while True:
                while_loop += 1
                # if while_loop%10000 == 0:
                #     print("while_loop:\t",while_loop)
                #     print("len(stack):\t",len(stack))
                #     print("arrived_sum:\t",sum(arrived))
                #     print("arrived_num:\t",arrived_num)
                #     print(label_num)

                # if arrived_num%100 == 0:
                #     print("len(stack):\t",len(stack))
                #     print("arrived_sum:\t",sum(arrived))
                #     print("arrived_num:\t",arrived_num)
                #     print(label_num)
                neighb = neighborhoods[i]

                arrived_num += 1
                if labels[i] == -1 :
                    neighb_labels = tuple(labels[v] for v in neighb if labels[v] != -1)
                    if len(neighb_labels) > self.min_dense:
                        labels_counter = Counter(neighb_labels)
                        label_num, counter = labels_counter.most_common(1)[0]
                        if counter > self.min_dense:
                            labels[i] = label_num

                for v in neighb:
                    if not arrived[v]:
                        arrived[v] = True
                        stack.append(v)

                if len(stack) == 0:
                    break
                i = stack.popleft()
        #     print("finish while")
        # print("tt arrived_sum:\t",sum(arrived))
        # print("tt arrived_num:\t",arrived_num)
        # print(label_num)
        return labels
