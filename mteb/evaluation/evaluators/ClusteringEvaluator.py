import logging

import numpy as np
import sklearn
import sklearn.cluster

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, clustering_batch_size=500, batch_size=32, limit=None, clustering_algorithm='k-means', **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.batch_size = batch_size
        self.clustering_algorithm = clustering_algorithm

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        corpus_embeddings = np.asarray(model.encode(self.sentences, batch_size=self.batch_size))

        logger.info(f"Fitting {self.clustering_algorithm} model...")
        K = len(set(self.labels))
        if self.clustering_algorithm == 'k-means':
            clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=K, batch_size=self.clustering_batch_size, n_init="auto"
            )
        elif self.clustering_algorithm == 'spherical-k-means':
            from spherecluster import SphericalKMeans
            clustering_model = SphericalKMeans(n_clusters=K)
        elif self.clustering_algorithm == 'vMF-mixture-soft':
            from spherecluster import VonMisesFisherMixture
            clustering_model = VonMisesFisherMixture(n_clusters=K, posterior_type='soft')
        elif self.clustering_algorithm == 'vMF-mixture-hard':
            from spherecluster import VonMisesFisherMixture
            clustering_model = VonMisesFisherMixture(n_clusters=K, posterior_type='hard')
        else:
            raise NotImplementedError

        if self.clustering_algorithm != 'k-means':
            assert corpus_embeddings.ndim == 2
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=-1)

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}
