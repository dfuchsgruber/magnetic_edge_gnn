import random

import numpy as np

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_inductive,
)

from .inductive_dataset import InductiveDataset


class RandomWalkDenoisingDataset(InductiveDataset):
    """
    Dataset class for the synthetic random walk denoising task.
    Given a directed graph with edge transition probabilities and a noisy random walk, the goal is to recover the complete random walk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, add_zeros_to_flow_input=True)
    


    def preprocess(self):
        num_nodes = 50
        num_edges = 200
        num_graphs = 1000
        random_walk_length = 100
        p_show_label = 0.20

        p_edge = num_edges / (num_nodes * (num_nodes - 1))

        all_equi_features = []
        all_inv_features = []
        all_undirected_edges = []
        all_labels = []
        for _ in range(num_graphs):
            cur_random_walk_length = 0

            while cur_random_walk_length < random_walk_length - 1:
                adjacency_matrix = np.random.uniform(
                    low=0.0, high=1.0, size=(num_nodes, num_nodes)
                )
                np.fill_diagonal(adjacency_matrix, 0.0)

                keep_edges = (
                    np.random.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
                    < p_edge
                )
                adjacency_matrix[~keep_edges] = 0

                row, column = np.nonzero(adjacency_matrix)
                equi_features = {(u, v): np.zeros(0) for u, v in zip(row, column)}
                # Features: [noisy whether the edge was traversed in the random walk, transition probability]
                inv_features = {
                    (u, v): np.array([0, adjacency_matrix[u, v]])
                    for u, v in zip(row, column)
                }
                undirected_edges = {(u, v): 0 for u, v in zip(row, column)}
                labels = {(u, v): 0 for u, v in zip(row, column)}

                cur_node = np.random.randint(low=0, high=num_nodes, size=1)[0]
                for cur_random_walk_length in range(random_walk_length):
                    # If there are no outgoing edges, finish the random walk.
                    if adjacency_matrix[cur_node, :].sum() == 0:
                        break

                    p_transition = (
                        adjacency_matrix[cur_node, :]
                        / adjacency_matrix[cur_node, :].sum()
                    )
                    next_node = np.random.choice(num_nodes, p=p_transition)
                    labels[(cur_node, next_node)] = 1
                    cur_node = next_node

            # With ´p_show_label´ probability show the actual label.
            for e in inv_features:
                if random.random() < p_show_label:
                    inv_features[e][0] = labels[e]

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
