import random

import numpy as np

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_inductive,
)

from .inductive_dataset import InductiveDataset


class MixedLongestCycleIdentificationDataset(InductiveDataset):
    """
    Dataset class for the synthetic longest cycle identification task in graphs with directed and undirected edges.
    Given a graph with directed and undirected edges, the goal is to identify which edges are lying on the longest directed cycle.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, add_zeros_to_flow_input=True)
    


    def preprocess(self):
        min_num_nodes_per_cycle = 6
        max_num_nodes_per_cycle = 8
        average_degree = 2
        p_undirected = 0.25
        num_graphs = 1000

        all_equi_features = []
        all_inv_features = []
        all_undirected_edges = []
        all_labels = []
        for _ in range(num_graphs):
            num_nodes_per_cycle = random.randint(
                min_num_nodes_per_cycle, max_num_nodes_per_cycle
            )
            num_edges_per_cycle = num_nodes_per_cycle * average_degree

            # Construction of the longest directed cycle.
            cycle_edges = []
            for i in range(num_nodes_per_cycle):
                u = i
                v = (i + 1) % num_nodes_per_cycle
                cycle_edges.append((u, v))

            all_edges_cycle = [
                (u, v)
                for u in range(num_nodes_per_cycle)
                for v in range(num_nodes_per_cycle)
                if u != v
            ]

            # We remove (0, num_nodes_per_cycle - 1) edge so it is impossible to get a full cycle in the reverse direction.
            random_edges_cycle = list(
                set(all_edges_cycle)
                - set(cycle_edges)
                - set((0, num_nodes_per_cycle - 1))
            )
            random_edges_cycle = random.sample(
                random_edges_cycle, num_edges_per_cycle - num_nodes_per_cycle
            )

            # Construction of the undirected cycle.
            undirected_cycle_edges = []
            for i in range(num_nodes_per_cycle):
                u = i + num_nodes_per_cycle
                v = (i + 1) % num_nodes_per_cycle + num_nodes_per_cycle
                undirected_cycle_edges.append((u, v))

            all_edges_undirected_cycle = [
                (u + num_nodes_per_cycle, v + num_nodes_per_cycle)
                for u in range(num_nodes_per_cycle)
                for v in range(num_nodes_per_cycle)
                if u != v
            ]

            # We remove (num_nodes_per_cycle, 2 * num_nodes_per_cycle - 1) edge so it is impossible to get a full cycle in the reverse direction.
            random_edges_undirected_cycle = list(
                set(all_edges_undirected_cycle)
                - set(undirected_cycle_edges)
                - set((num_nodes_per_cycle, 2 * num_nodes_per_cycle - 1))
            )
            random_edges_undirected_cycle = random.sample(
                random_edges_undirected_cycle, num_edges_per_cycle - num_nodes_per_cycle
            )

            # Connect both cycles.
            random_edges_cycle.append((0, num_nodes_per_cycle))

            equi_features = {}
            inv_features = {}
            undirected_edges = {}
            labels = {}
            for e in cycle_edges:
                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                undirected_edges[e] = 0
                labels[e] = 1

            for idx, e in enumerate(undirected_cycle_edges):
                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                # Make a single edge on the cycle undirected.
                undirected_edges[e] = 1 if idx == 0 else 0
                labels[e] = 0

            for e in random_edges_cycle + random_edges_undirected_cycle:
                equi_features[e] = np.zeros(0)
                inv_features[e] = np.ones(1)
                undirected_edges[e] = 1 if random.random() < p_undirected else 0
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
