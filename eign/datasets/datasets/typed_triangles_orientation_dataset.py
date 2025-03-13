import random

import numpy as np
from torch_geometric.data import Data

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_inductive,
)

from .inductive_dataset import InductiveDataset


class TypedTrianglesOrientationDataset(InductiveDataset):
    """
    Dataset class for the synthetic typed triangles orientation task.
    Given a graph with edge types and flows, for each edge that lies on a directed triangle with edges of the same type,
    orient its flow in the correct direction, otherwise output 0.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, add_zeros_to_flow_input=True)
    

    @staticmethod
    def create_directed_edge(edge: tuple[int, int], edge_type: int, label: int):
        weight = 1
        undirected_edge = 0
        return edge, weight, edge_type, label, undirected_edge

    @staticmethod
    def create_arbitrary_edge(
        edge: tuple[int, int], edge_type: int, label: int, p_undirected: float
    ):
        if random.random() < p_undirected:
            # Undirected edge.
            weight = 1 if random.random() < 0.5 else -1
            undirected_edge = 1

            # Randomly swap the orientation.
            if random.random() < 0.5:
                edge = (edge[1], edge[0])
                weight = -weight
                label = -label
        else:
            # Directed edge.
            weight = 1
            undirected_edge = 0

        return edge, weight, edge_type, label, undirected_edge

    def preprocess(self) -> list[Data]:
        num_nodes = 300
        num_edges = 400
        p_undirected = 0.5
        num_types = 3
        num_graphs = 100

        all_equi_features = []
        all_inv_features = []
        all_undirected_edges = []
        all_labels = []
        for _ in range(num_graphs):
            equi_features = {}
            inv_features = {}
            undirected_edges = {}
            labels = {}

            used_edges = []
            # Triangles of different sorts.
            for i in range(num_nodes // 3):
                # Triangle types:
                # - 0: Directed triangle with edges of the same type.
                # - 1: At least one directed edge with the wrong direction, edges have the same type.
                # - 2: Directed triangle with an edge of the wrong type.
                triangle_type_p = np.array([0.5, 0.25, 0.25])
                triangle_type = np.random.choice(3, 1, p=triangle_type_p)[0]

                correct_edge_type = random.randint(0, num_types - 1)

                wrong_edge_type = correct_edge_type
                while wrong_edge_type == correct_edge_type:
                    wrong_edge_type = random.randint(0, num_types - 1)

                for j in range(3):
                    u = 3 * i + j
                    v = 3 * i + (j + 1) % 3
                    used_edges.append((u, v))
                    used_edges.append((v, u))

                    if triangle_type == 0:
                        # The first edge is always directed.
                        if j == 0:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_directed_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=1,
                                )
                            )
                        else:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_arbitrary_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=1,
                                    p_undirected=p_undirected,
                                )
                            )

                    if triangle_type == 1:
                        # The first edge is always directed in the correct direction,
                        # the second edge is always directed in the wrong direction.
                        if j == 0:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_directed_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=0,
                                )
                            )
                        elif j == 1:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_directed_edge(
                                    edge=(v, u),
                                    edge_type=correct_edge_type,
                                    label=0,
                                )
                            )
                        else:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_arbitrary_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=0,
                                    p_undirected=p_undirected,
                                )
                            )

                    if triangle_type == 2:
                        # The first edge is always directed,
                        # the third edge has always the wrong type.
                        if j == 0:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_directed_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=0,
                                )
                            )
                        elif j == 1:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_arbitrary_edge(
                                    edge=(u, v),
                                    edge_type=correct_edge_type,
                                    label=0,
                                    p_undirected=p_undirected,
                                )
                            )
                        else:
                            edge, weight, edge_type, label, undirected_edge = (
                                self.create_arbitrary_edge(
                                    edge=(u, v),
                                    edge_type=wrong_edge_type,
                                    label=0,
                                    p_undirected=p_undirected,
                                )
                            )

                    equi_features[edge] = np.array([weight])
                    one_hot_edge_type = np.zeros(num_types)
                    one_hot_edge_type[edge_type] = 1
                    inv_features[edge] = one_hot_edge_type
                    undirected_edges[edge] = undirected_edge
                    labels[edge] = label

            # Edges that connect triangles and are not part of any triangle.
            for i in range(num_nodes // 3 - 1):
                if len(equi_features) == num_edges:
                    break

                u = 3 * i + random.randint(0, 2)
                v = 3 * (i + 1) + random.randint(0, 2)

                if (u, v) in used_edges or (v, u) in used_edges:
                    continue

                used_edges.append((u, v))
                used_edges.append((v, u))

                random_edge_type = random.randint(0, num_types - 1)
                edge, weight, edge_type, label, undirected_edge = (
                    self.create_arbitrary_edge(
                        edge=(u, v),
                        edge_type=random_edge_type,
                        label=0,
                        p_undirected=p_undirected,
                    )
                )

                equi_features[edge] = np.array([weight])
                one_hot_edge_type = np.zeros(num_types)
                one_hot_edge_type[edge_type] = 1
                inv_features[edge] = one_hot_edge_type
                undirected_edges[edge] = undirected_edge
                labels[edge] = label

            all_edges = [
                (u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v
            ]
            random_edges = list(set(all_edges) - set(used_edges))
            random.shuffle(random_edges)
            # Random edges that may have wrong labels!
            for u, v in random_edges:
                if len(equi_features) == num_edges:
                    break

                if (u, v) in used_edges or (v, u) in used_edges:
                    continue

                used_edges.append((u, v))
                used_edges.append((v, u))

                random_edge_type = random.randint(0, num_types - 1)
                edge, weight, edge_type, label, undirected_edge = (
                    self.create_arbitrary_edge(
                        edge=(u, v),
                        edge_type=random_edge_type,
                        label=0,
                        p_undirected=p_undirected,
                    )
                )

                equi_features[edge] = np.array([weight])
                one_hot_edge_type = np.zeros(num_types)
                one_hot_edge_type[edge_type] = 1
                inv_features[edge] = one_hot_edge_type
                undirected_edges[edge] = undirected_edge
                labels[edge] = label

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
