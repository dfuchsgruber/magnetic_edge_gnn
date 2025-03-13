import random

import numpy as np

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_inductive,
)

from .inductive_dataset import InductiveDataset


class DAGDenoisingDataset(InductiveDataset):
    """
    Dataset class for the synthetic directed acyclic graph (DAG) denoising task.
    Given a DAG where the directions of some edges are flipped, the goal is to identify which ones have been flipped.
    """

    def preprocess(self):
        num_nodes = 50
        num_edges = 500
        num_graphs = 1000
        p_flip = 0.20

        all_equi_features = []
        all_inv_features = []
        all_undirected_edges = []
        all_labels = []
        for _ in range(num_graphs):
            all_edges = [(u, v) for u in range(1, num_nodes) for v in range(u)]
            edges = random.sample(all_edges, num_edges)

            equi_features = {}
            inv_features = {}
            undirected_edges = {}
            labels = {}
            for e in edges:
                label = 0

                # Flip the edge with probability p_flip.
                if random.random() < p_flip:
                    e = (e[1], e[0])
                    label = 1

                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                undirected_edges[e] = 0
                labels[e] = label

            all_equi_features.append(equi_features)
            all_inv_features.append(inv_features)
            all_undirected_edges.append(undirected_edges)
            all_labels.append(labels)

        # Create PyG graphs from the dictonaries.
        data = create_pyg_graph_inductive(
            all_equi_features=all_equi_features,
            all_inv_features=all_inv_features,
            all_undirected_edges=all_undirected_edges,
            all_labels=all_labels,
            max_num_positional_laplacian_encodings=self.max_num_positional_laplacian_encodings,
            laplacian_encodings_phase_shift=self.laplacian_encodings_phase_shift,
        )
        return data