import random

import numpy as np

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_inductive,
)

from .inductive_dataset import InductiveDataset


class LongestCycleIdentificationDataset(InductiveDataset):
    """
    Dataset class for the synthetic longest cycle identification task.
    Given a directed graph, the goal is to identify which edges are lying on the longest directed cycle.
    """


    def preprocess(self):
        num_nodes = 6
        num_edges = 12
        num_graphs = 1000

        all_equi_features = []
        all_inv_features = []
        all_undirected_edges = []
        all_labels = []
        for _ in range(num_graphs):
            cycle_edges = []
            for i in range(num_nodes):
                u = i
                v = (i + 1) % num_nodes
                cycle_edges.append((u, v))

            all_edges = [
                (u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v
            ]

            # We remove (0, num_nodes - 1) edge so it is impossible to get a full cycle in the reverse direction.
            random_edges = list(
                set(all_edges) - set(cycle_edges) - set((0, num_nodes - 1))
            )
            random_edges = random.sample(random_edges, num_edges - num_nodes)

            equi_features = {}
            inv_features = {}
            undirected_edges = {}
            labels = {}
            for e in cycle_edges:
                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                undirected_edges[e] = 0
                labels[e] = 1

            for e in random_edges:
                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                undirected_edges[e] = 0
                labels[e] = 0

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
