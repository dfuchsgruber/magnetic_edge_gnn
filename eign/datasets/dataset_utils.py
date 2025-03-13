"""Utility functions for creating and splitting PyTorch Geometric datasets."""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg._eigen.arpack import ArpackError
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from verstack.stratified_continuous_split import scsplit

from magnetic_edge_gnn.models.model_utils import magnetic_edge_laplacian


def random_orientation(
    data: Data,
    orientation_equivariant_labels: bool,
    reorient_directed_edges: bool = False,
    seed: int | None = None,
) -> Data:
    """
    Randomly orients the edges in the edge list.

    Args:
        data (Data): PyTorch Geometric data object.
        orientation_equivariant_labels (bool): Whether the labels are orientation-equivariant or not.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        Data: PyTorch Geometric data object with randomly oriented edges.
    """
    edge_index, equi_edge_attr, y = (
        data.edge_index,
        data.equi_edge_attr,
        data.y,
    )
    # create a random torch mask based on the seed on a separate rng
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.manual_seed(torch.randint(0, 2**32, (1,)).item())

    swap_edges = (
        torch.rand(size=(edge_index.shape[1],), device=edge_index.device, generator=rng)
        < 0.5
    )
    if not reorient_directed_edges:
        swap_edges = swap_edges & data.undirected_mask

    new_edge_index = edge_index.clone().detach()
    new_edge_index[:, swap_edges] = torch.stack(
        [
            edge_index[1, swap_edges],
            edge_index[0, swap_edges],
        ]
    )

    # Flip sign for all orientation-equivariant features and orientation-equivariant labels.
    equi_edge_attr[swap_edges] = -equi_edge_attr[swap_edges]
    if orientation_equivariant_labels:
        y[swap_edges] = -y[swap_edges]

    data.update(
        dict(
            edge_index=new_edge_index,
            equi_edge_attr=equi_edge_attr,
            y=y,
        )
    )

    return data


def combine_edges(features: dict, flows: dict) -> tuple[dict, dict, dict]:
    """
    Combine directed edges with reversed directions to form undirected edges.

    Args:
        features (dict): Features of the edges.
        flows (dict): Flows of the edges.

    Returns:
        tuple[dict, dict, dict]:
            Combined features, combined flows, and undirected edges.
    """
    combined_features = {}
    combined_flows = {}
    undirected_edges = {}

    for e in features.keys():
        u, v = e
        if (v, u) in features.keys():
            # Undirected edge. Add the undirected edge only if it has not been added yet.
            if (v, u) not in combined_features:
                combined_features[(u, v)] = (features[(u, v)] + features[(v, u)]) / 2
                undirected_edges[(u, v)] = 1

                if (u, v) in flows or (v, u) in flows:
                    flow_1 = flows[(u, v)] if (u, v) in flows else 0
                    flow_2 = flows[(v, u)] if (v, u) in flows else 0
                    combined_flows[(u, v)] = flow_1 - flow_2
        else:
            # Directed edge.
            combined_features[e] = features[e]
            undirected_edges[e] = 0

            if e in flows:
                combined_flows[e] = flows[e]

    return combined_features, combined_flows, undirected_edges


def relabel_nodes(
    features: dict, flows: dict, undirected_edges: dict
) -> tuple[dict, dict, dict, dict]:
    """
    Relabels nodes to 0, ..., N - 1 range and updates the edges.

    Args:
        features (dict): Features of the edges.
        flows (dict): Flows of the edges.
        undirected_edges (dict): Whether the edges are undirected (label 1) or directed (label 0).

    Returns:
        tuple[dict, dict, dict, dict]:
            Relabeled features, relabeled flows, relabeled undirected edges, and node mapping.
    """
    all_nodes = [node for edge in features.keys() for node in edge]
    node_mapping = {node: idx for idx, node in enumerate(set(all_nodes))}

    relabeled_features = {
        (node_mapping[u], node_mapping[v]): feat for ((u, v), feat) in features.items()
    }
    relabeled_flows = {
        (node_mapping[u], node_mapping[v]): flow for ((u, v), flow) in flows.items()
    }
    relabeled_undirected_edges = {
        (node_mapping[u], node_mapping[v]): undirected_edge
        for ((u, v), undirected_edge) in undirected_edges.items()
    }

    return relabeled_features, relabeled_flows, relabeled_undirected_edges, node_mapping


def normalize_flows(
    features: dict,
    flows: dict,
    undirected_edges: dict,
    normalize_by_max_flow: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Converts flow estimation instance to a non-negative one for directed edges and normalizes all values.

    Args:
        features (dict): Features of the edges.
        flows (dict): Flows of the edges.
        undirected_edges (dict): Whether the edges are undirected (label 1) or directed (label 0).

    Returns:
        tuple[dict, dict, dict]: New features, normalized flows, and new undirected edges.
    """

    normalized_flows = {}
    new_features = {}
    new_undirected_edges = {}

    if normalize_by_max_flow:
        max_flow = max([abs(f) for f in flows.values()])
    else:
        max_flow = 1.0
    for edge in features.keys():
        if edge in flows:
            # Flip only directed edges with flows against their direction.
            if undirected_edges[edge] == 0:
                if flows[edge] < 0:
                    flipped_edge = (edge[1], edge[0])
                    normalized_flows[flipped_edge] = -flows[edge] / max_flow
                    new_features[flipped_edge] = features[edge]
                    new_undirected_edges[flipped_edge] = undirected_edges[edge]
                else:
                    normalized_flows[edge] = flows[edge] / max_flow
                    new_features[edge] = features[edge]
                    new_undirected_edges[edge] = undirected_edges[edge]
            else:
                # Randomly flip the undirected edges.
                if random.random() < 0.5:
                    flipped_edge = (edge[1], edge[0])
                    normalized_flows[flipped_edge] = -flows[edge] / max_flow
                    new_features[flipped_edge] = features[edge]
                    new_undirected_edges[flipped_edge] = undirected_edges[edge]
                else:
                    normalized_flows[edge] = flows[edge] / max_flow
                    new_features[edge] = features[edge]
                    new_undirected_edges[edge] = undirected_edges[edge]
        else:
            new_features[edge] = features[edge]
            new_undirected_edges[edge] = undirected_edges[edge]

    return new_features, normalized_flows, new_undirected_edges


def normalize_features(features: dict) -> tuple[dict, dict]:
    """
    Normalizes features using standard scaler.

    Args:
        features (dict): Features of the edges.

    Returns:
        dict: Normalized features.
    """
    scaler = StandardScaler()

    num_features = features[next(iter(features))].shape[0]
    feature_matrix = np.zeros((len(features), num_features))

    for i, e in enumerate(features):
        feature_matrix[i] = features[e]

    feature_matrix = scaler.fit_transform(feature_matrix)

    normalized_features = {}
    for i, e in enumerate(features):
        normalized_features[e] = feature_matrix[i]

    return normalized_features


def continuous_idx_split(
    values,
    train_size: float,
    val_size: float,
    test_size: float | None = None,
    random_state: int | None = None,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a stratified split of the indices based on continuous target values."""
    data = pd.DataFrame({"y": values})
    if test_size is None:
        test_size = 1 - train_size - val_size

    if stratify:
        train_val, test = scsplit(
            data,
            stratify=data["y"],
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=random_state,
        )
        train, val = scsplit(
            train_val,
            stratify=train_val["y"],
            train_size=train_size / (train_size + val_size),
            test_size=val_size / (train_size + val_size),
            random_state=random_state,
        )
        idx_train = np.array(train.index)
        idx_val = np.array(val.index)
        idx_test = np.array(test.index)

    else:
        idxs = np.arange(len(data))
        rng = np.random.default_rng(random_state)
        rng.shuffle(idxs)
        idx_train = idxs[: int(train_size * len(data))]
        idx_val = idxs[
            int(train_size * len(data)) : int((train_size + val_size) * len(data))
        ]
        idx_test = idxs[int((train_size + val_size) * len(data)) :]

    assert (
        len(np.intersect1d(idx_train, idx_val)) == 0
        and len(np.intersect1d(idx_train, idx_test)) == 0
        and len(np.intersect1d(idx_val, idx_test)) == 0
    )
    return idx_train, idx_val, idx_test


def create_pyg_graph_transductive(
    equi_features: dict,
    inv_features: dict,
    undirected_edges: dict,
    labels: dict,
    max_num_positional_laplacian_encodings: int = 0,
    laplacian_encodings_phase_shift: float = 0.0,
    train_edges: dict | None = None,
) -> Data:
    """
    Creates a PyTorch Geometric data object based on features and labels.
    It also creates a random transductive split of the labeled edges into train/validation/test.

    Args:
        equi_features (dict): Orientation-equivariant features of the edges.
        inv_features (dict): Orientation-invariant features of the edges.
        undirected_edges (dict): Whether the edges are undirected (label 1) or directed (label 0).
        labels (dict): Labels of the edges.
        max_num_positional_laplacian_encodings (int, optional): Number of positional Laplacian encodings. Defaults to 0.
        laplacian_encodings_phase_shift (float, optional): Phase shift of the Laplacian encodings. Defaults to 0.0.
        train_edges (dict, optional): Fixed train edges. Defaults to None.

    Returns:
        Data: PyTorch Geometric data object.
    """
    num_edges = len(equi_features)
    if train_edges is None:
        train_edges = {}

    num_equi_features = equi_features[list(equi_features.keys())[0]].shape[0]
    equi_edge_attr = torch.zeros(num_edges, num_equi_features)
    num_inv_features = inv_features[list(inv_features.keys())[0]].shape[0]
    inv_edge_attr = torch.zeros(num_edges, num_inv_features)

    edge_index = torch.zeros(2, num_edges, dtype=torch.long)
    undirected_mask = torch.zeros(num_edges, dtype=bool)
    y = torch.zeros(num_edges, dtype=torch.float)
    labeled_mask = torch.zeros(num_edges, dtype=bool)
    train_mask = torch.zeros(num_edges, dtype=bool)

    for i, e in enumerate(equi_features):
        edge_index[:, i] = torch.tensor([e[0], e[1]])
        equi_edge_attr[i] = torch.tensor(equi_features[e])
        inv_edge_attr[i] = torch.tensor(inv_features[e])
        undirected_mask[i] = undirected_edges[e]

        if e in labels:
            y[i] = labels[e]
            labeled_mask[i] = True
        if e in train_edges:
            train_mask[i] = True

    num_nodes = int(edge_index.max().item()) + 1

    return Data(
        num_nodes=num_nodes,
        edge_index=edge_index,
        equi_edge_attr=equi_edge_attr,
        inv_edge_attr=inv_edge_attr,
        undirected_mask=undirected_mask,
        y=y,
        graph_idx=str(0),
        labeled_mask=labeled_mask,
        train_mask=train_mask,
        **get_laplacian_eigenvectors(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            q=laplacian_encodings_phase_shift,
            k=max_num_positional_laplacian_encodings,
        ),
    )


def augment_graph_equivariant_inputs(
    data: Data,
    add_noisy_flow_to_input: bool = False,
    add_interpolation_flow_to_input: bool = False,
    add_zeros_to_flow_input: bool = False,
    interpolation_label_size: float = 0.75,
) -> Data:
    """Augments the equivariant inputs of a graph.

    Args:
        data (Data): PyTorch Geometric data object.
        add_noisy_flow_to_input (bool, optional): If noisy flows should be added to the input. Defaults to False.
        add_interpolation_flow_to_input (bool, optional): If interpolation flows should be added to the input. Defaults to False.
        add_zeros_to_flow_input (bool, optional): If zeros should be added to the input. Defaults to False.
        interpolation_label_size (float, optional): How many interpolation labels should be added. Defaults to 0.75.

    Returns:
        Data: PyTorch Geometric data object with augmented equivariant inputs.
    """
    y = data.y
    equi_edge_attr = data.equi_edge_attr
    num_edges = data.equi_edge_attr.size(0)

    # Add noisy flow as an orientation-equivariant feature.
    if add_noisy_flow_to_input:
        y_std = torch.std(y).item()
        # Add noise from Uniform(-y_std, y_std).
        l_noise, u_noise = -y_std, y_std
        noisy_flows = y + ((torch.rand(num_edges) * (l_noise - u_noise)) + u_noise)
        equi_edge_attr = torch.cat([noisy_flows.reshape(-1, 1), equi_edge_attr], dim=-1)

    # Add flow as an orientation-equivariant feature to 75% of the training samples.
    if add_interpolation_flow_to_input:
        interpolation_flows = torch.zeros(num_edges)
        interpolation_flows_mask = (
            torch.rand((num_edges)) < interpolation_label_size
        ) & data.train_mask
        interpolation_flows[interpolation_flows_mask] = y[interpolation_flows_mask]
        equi_edge_attr = torch.cat(
            [interpolation_flows.reshape(-1, 1), equi_edge_attr], dim=-1
        )

        # Remove the samples with given labels from the training mask.
        data.train_mask[interpolation_flows_mask] = False

    if add_zeros_to_flow_input:
        equi_edge_attr = torch.cat([torch.zeros(num_edges, 1), equi_edge_attr], dim=-1)

    data.equi_edge_attr = equi_edge_attr
    return data


def split_pyg_graph_transductive(
    graphs: list[Data],
    val_ratio: float,
    test_ratio: float,
    stratified_split: bool = False,
    seed: int | None = None,
    interpolation_label_size: float = 0.75,
    add_noisy_flow_to_input: bool = False,
    add_interpolation_flow_to_input: bool = False,
    add_zeros_to_flow_input: bool = False,
) -> list[Data]:
    """Splits the graphs into train/validation/test sets for transductive problems.

    Args:
        graphs (list[Data]): List of PyTorch Geometric data objects.
        val_ratio (float): Ratio of validation samples.
        test_ratio (float): Ratio of test samples.
        stratified_split (bool, optional): Whether to perform stratified split. Defaults to False.
        seed (int, optional): Random seed. Defaults to None.
        interpolation_label_size (float, optional): How many interpolation labels should be added. Defaults to 0.75.
        add_noisy_flow_to_input (bool, optional): If noisy flows should be added to the input. Defaults to False.
        add_interpolation_flow_to_input (bool, optional): If interpolation flows should be added to the input. Defaults to False.
        add_zeros_to_flow_input (bool, optional): If zeros should be added to the input. Defaults to False.
    
    Returns:
        list[Data]: List of PyTorch Geometric data objects with train/validation/test masks.
    """
    graphs_new = []
    val_ratio_, test_ratio_ = val_ratio, test_ratio

    for data in graphs:
        data = deepcopy(data)
        num_edges = data.equi_edge_attr.size(0)
        val_ratio, test_ratio = val_ratio_, test_ratio_

        train_mask, val_mask, test_mask = (
            torch.zeros(num_edges, dtype=bool),
            torch.zeros(num_edges, dtype=bool),
            torch.zeros(num_edges, dtype=bool),
        )

        if hasattr(data, "train_mask") and data.train_mask.sum(0) > 0:
            print("Splitting w.r.t. fixed train mask.")
            num_val = int(num_edges * val_ratio)
            num_test = int(num_edges * test_ratio)

            labeled_mask = data.labeled_mask & (~data.train_mask)
            val_ratio = float(num_val / data.labeled_mask.sum(0))
            test_ratio = float(num_test / data.labeled_mask.sum(0))
            train_mask |= data.train_mask
        else:
            labeled_mask = data.labeled_mask
        labeled_idx = torch.nonzero(labeled_mask, as_tuple=True)[0]

        idx_train, idx_val, idx_test = continuous_idx_split(
            data.y[labeled_idx],
            1 - val_ratio - test_ratio,
            val_ratio,
            test_ratio,
            seed,
            stratify=stratified_split,
        )

        train_mask[labeled_idx[idx_train]] = True
        val_mask[labeled_idx[idx_val]] = True
        test_mask[labeled_idx[idx_test]] = True
        assert (
            not torch.any(train_mask & val_mask)
            and not torch.any(val_mask & test_mask)
            and not torch.any(train_mask & test_mask)
        )
        assert (
            # not (train_mask & ~(labeled_mask)).any() # we do not check against train, as the `data.train_mask` attribute may force us to have this
            not (val_mask & ~(labeled_mask)).any()
            and not (test_mask & ~(labeled_mask)).any()
        )

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data = augment_graph_equivariant_inputs(
            data,
            add_noisy_flow_to_input=add_noisy_flow_to_input,
            add_interpolation_flow_to_input=add_interpolation_flow_to_input,
            add_zeros_to_flow_input=add_zeros_to_flow_input,
            interpolation_label_size=interpolation_label_size,
        )

        graphs_new.append(data)

    return graphs_new


def create_pyg_graph_inductive(
    all_equi_features: list[dict],
    all_inv_features: list[dict],
    all_undirected_edges: list[dict],
    all_labels: list[dict],
    max_num_positional_laplacian_encodings: int = 0,
    laplacian_encodings_phase_shift: float = 0.0,
) -> list[Data]:
    """Creates a list of PyTorch Geometric data objects based on features and labels.
    
    Args:
        all_equi_features (list[dict]): List of orientation-equivariant features of the edges.
        all_inv_features (list[dict]): List of orientation-invariant features of the edges.
        all_undirected_edges (list[dict]): List of whether the edges are undirected (label 1) or directed (label 0).
        all_labels (list[dict]): List of labels of the edges.
        max_num_positional_laplacian_encodings (int, optional): Number of positional Laplacian encodings. Defaults to 0.
        laplacian_encodings_phase_shift (float, optional): Phase shift of the Laplacian encodings. Defaults to 0.0.
        
    Returns:
        list[Data]: List of PyTorch Geometric data objects.
    """
    
    dataset = []
    for graph_idx, (equi_features, inv_features, labels, undirected_edges) in enumerate(
        zip(all_equi_features, all_inv_features, all_labels, all_undirected_edges)
    ):
        num_edges = len(equi_features)

        num_equi_features = equi_features[list(equi_features.keys())[0]].shape[0]
        equi_edge_attr = torch.zeros(num_edges, num_equi_features)
        num_inv_features = inv_features[list(inv_features.keys())[0]].shape[0]
        inv_edge_attr = torch.zeros(num_edges, num_inv_features)

        edge_index = torch.zeros(2, num_edges, dtype=torch.long)
        undirected_mask = torch.zeros(num_edges, dtype=bool)
        y = torch.zeros(num_edges, dtype=torch.float)

        for i, e in enumerate(equi_features):
            edge_index[:, i] = torch.tensor([e[0], e[1]])
            equi_edge_attr[i] = torch.tensor(equi_features[e])
            inv_edge_attr[i] = torch.tensor(inv_features[e])
            undirected_mask[i] = undirected_edges[e]

            if e in labels:
                y[i] = labels[e]

        num_nodes = int(edge_index.max().item()) + 1

        data = Data(
            num_nodes=num_nodes,
            edge_index=edge_index,
            equi_edge_attr=equi_edge_attr,
            inv_edge_attr=inv_edge_attr,
            undirected_mask=undirected_mask,
            y=y,
            graph_idx=str(graph_idx),
            **get_laplacian_eigenvectors(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                q=laplacian_encodings_phase_shift,
                k=max_num_positional_laplacian_encodings,
            ),
        )
        dataset.append(data)
    return dataset


def split_pyg_graph_inductive(
    graphs: list[Data],
    val_ratio: float,
    test_ratio: float,
    interpolation_label_size: float = 0.75,
    add_noisy_flow_to_input: bool = False,
    add_interpolation_flow_to_input: bool = False,
    add_zeros_to_flow_input: bool = False,
    seed: int | None = None,
) -> dict[str, list[Data]]:
    """Splits the graphs into train/validation/test sets for inductive problems.
    
    Args:
        graphs (list[Data]): List of PyTorch Geometric data objects.
        val_ratio (float): Ratio of validation samples.
        test_ratio (float): Ratio of test samples.
        interpolation_label_size (float, optional): How many interpolation labels should be added. Defaults to 0.75.
        add_noisy_flow_to_input (bool, optional): If noisy flows should be added to the input. Defaults to False.
        add_interpolation_flow_to_input (bool, optional): If interpolation flows should be added to the input. Defaults to False.
        add_zeros_to_flow_input (bool, optional): If zeros should be added to the input. Defaults to False.
        seed (int, optional): Random seed. Defaults to None.
        
    Returns:
        dict[str, list[Data]]: Dictionary with train/validation/test sets.
    """
    # Split the graphs into train/validation/test sets.
    num_graphs = len(graphs)

    # get a random permutation based on seed argument
    perm = torch.randperm(
        num_graphs,
        generator=(torch.Generator().manual_seed(seed) if seed is not None else None),
    )
    val_idx = int(num_graphs * (1 - val_ratio - test_ratio))
    test_idx = int(num_graphs * (1 - test_ratio))

    train_graph_mask, val_graph_mask, test_graph_mask = (
        torch.zeros(num_graphs, dtype=bool),
        torch.zeros(num_graphs, dtype=bool),
        torch.zeros(num_graphs, dtype=bool),
    )
    train_graph_mask[perm[:val_idx]] = True
    val_graph_mask[perm[val_idx:test_idx]] = True
    test_graph_mask[perm[test_idx:]] = True

    assert (
        not torch.any(train_graph_mask & val_graph_mask)
        and not torch.any(val_graph_mask & test_graph_mask)
        and not torch.any(train_graph_mask & test_graph_mask)
    )

    dataset = {"train": [], "val": [], "test": []}
    for graph_idx, graph in enumerate(graphs):
        graph = deepcopy(graph)
        num_edges = graph.equi_edge_attr.size(0)
        train_mask, val_mask, test_mask = (
            torch.zeros(num_edges, dtype=bool),
            torch.zeros(num_edges, dtype=bool),
            torch.zeros(num_edges, dtype=bool),
        )
        if train_graph_mask[graph_idx]:
            train_mask = torch.ones(num_edges, dtype=bool)
        elif val_graph_mask[graph_idx]:
            val_mask = torch.ones(num_edges, dtype=bool)
        elif test_graph_mask[graph_idx]:
            test_mask = torch.ones(num_edges, dtype=bool)

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

        graph = augment_graph_equivariant_inputs(
            graph,
            add_noisy_flow_to_input=add_noisy_flow_to_input,
            add_interpolation_flow_to_input=add_interpolation_flow_to_input,
            add_zeros_to_flow_input=add_zeros_to_flow_input,
            interpolation_label_size=interpolation_label_size,
        )

        if train_graph_mask[graph_idx]:
            dataset["train"].append(graph)
        elif val_graph_mask[graph_idx]:
            dataset["val"].append(graph)
        elif test_graph_mask[graph_idx]:
            dataset["test"].append(graph)
    return dataset


def get_laplacian_eigenvectors(edge_index, undirected_mask, q: float, k: int) -> dict[str, torch.Tensor]:
    """Computes the eigenvectors of the magnetic edge Laplacians for the given graph.

    Args:
        edge_index (torch.Tensor): Edge index tensor, shape (2, num_edges).
        undirected_mask (torch.Tensor): Mask for undirected edges, shape (num_edges,).
        q (float): Phase shift of the Laplacian encodings.
        k (int): Number of eigenvectors to compute.

    Returns:
        dict[str, torch.Tensor]: Dictionary with the real and imaginary parts of the eigenvectors.
    """
    if k <= 0:
        return {}
    L_inv = (
        magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type="orientation-invariant",
            q=q,
        )
        .cpu()
        .numpy()
    )
    L_equi = (
        magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type="orientation-equivariant",
            q=q,
        )
        .cpu()
        .numpy()
    )
    # Transform into scipy sparse matrix
    L_inv = scipy.sparse.csr_matrix(L_inv)
    L_equi = scipy.sparse.csr_matrix(L_equi)

    eigvec_inv = sparse_eigenvectors(L_inv, k=k)
    eigvec_equi = sparse_eigenvectors(L_equi, k=k)
    eigvec_inv_real = eigvec_inv.real[:, :k]
    eigvec_equi_real = eigvec_equi.real[:, :k]
    eigvec_inv_imag = eigvec_inv.imag[:, :k]
    eigvec_equi_imag = eigvec_equi.imag[:, :k]

    return dict(
        eigvec_inv_real=torch.from_numpy(eigvec_inv_real).float(),
        eigvec_inv_imag=torch.from_numpy(eigvec_inv_imag).float(),
        eigvec_equi_real=torch.from_numpy(eigvec_equi_real).float(),
        eigvec_equi_imag=torch.from_numpy(eigvec_equi_imag).float(),
    )


def sparse_eigenvectors(sparse_matrix, k: int=32, nvc: int | None=None) -> np.ndarray:
    """Computes the eigenvectors of a sparse matrix.
    
    Args:
        sparse_matrix (scipy.sparse.csr_matrix): Sparse matrix.
        k (int, optional): Number of eigenvectors to compute. Defaults to 32.
        nvc (int, optional): Number of vectors to compute. Defaults to None.
    
    Returns:
        np.ndarray: Eigenvectors of the sparse matrix.
    """
    k_original = k
    if k >= sparse_matrix.shape[0] - 1:
        k = sparse_matrix.shape[0] - 2

    try:
        _, eigenvectors = eigsh(sparse_matrix, k=k, which="SM")
    except ArpackError:
        # Assure that the dimensions are not too large
        if sparse_matrix.shape[0] > 2000:
            raise RuntimeError(
                "Matrix is too large to compute eigenvectors. Consider setting a smaller nvc."
            )
        # Dense version
        eigvals, eigenvectors = np.linalg.eig(np.array(sparse_matrix.todense()))
        eigenvectors = eigenvectors[:, np.argsort(np.abs(eigvals))][:, :k]
    # Except also runtime errors
    except RuntimeError as e:
        # Do just ones
        eigenvectors = np.ones((sparse_matrix.shape[0], k))

    if eigenvectors.shape[1] < k_original:
        eigenvectors = np.hstack(
            [eigenvectors, np.ones((sparse_matrix.shape[0], k_original - k))]
        )

    return eigenvectors
