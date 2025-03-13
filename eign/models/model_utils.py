import numpy as np
import torch
import torch.nn as nn


def activation_resolver(activation_name: str) -> callable:
    """
    Returns the activation function based on its name.

    Args:
        activation_name (str): Name of the activation function.

    Returns:
        callable: Activation function.
    """
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    else:
        ValueError(f"The activation function {activation_name} is not supported!")


def incidence_matrix(
    edge_index: torch.Tensor,
    undirected_mask: torch.Tensor,
    matrix_type: str,
    q: float = 0.0,
) -> torch.Tensor:
    """
    Computes the incidence matrix based on the edge list.

    Args:
        edge_index (torch.Tensor): Edge list.
        undirected_mask (torch.Tensor): Whether the edges are undirected (label True) or directed (label False).
        matrix_type (str): Whether the matrix is orientation-equivariant or orientation-invariant.
        q (float, optional): Potential for the Magnetic Edge Laplacian. Defaults to 0.0.

    Returns:
        torch.Tensor: Incidence matrix.
    """
    assert matrix_type in ["orientation-equivariant", "orientation-invariant"]

    num_nodes = int(edge_index.max().item()) + 1
    num_edges = edge_index.shape[1]
    B = torch.zeros(
        size=(num_nodes, num_edges),
        device=edge_index.device,
        dtype=torch.float if q == 0.0 else torch.cfloat,
    )

    tails = edge_index[0].int()
    heads = edge_index[1].int()
    edge_idxs = torch.arange(num_edges, dtype=int, device=edge_index.device)

    # Undirected edges.
    undirected_heads, undirected_tails, undirected_edge_idxs = (
        heads[undirected_mask == True],
        tails[undirected_mask == True],
        edge_idxs[undirected_mask == True],
    )

    # Directed edges.
    directed_heads, directed_tails, directed_edge_idxs = (
        heads[undirected_mask == False],
        tails[undirected_mask == False],
        edge_idxs[undirected_mask == False],
    )

    if matrix_type == "orientation-equivariant":
        # Incidence matrix contains "1"s for the heads of the undirected edges (*, v).
        B[undirected_heads, undirected_edge_idxs] = 1
        # Incidence matrix contains "-1"s for the tails of the undirected edges (v, *).
        B[undirected_tails, undirected_edge_idxs] = -1
    elif matrix_type == "orientation-invariant":
        # Incidence matrix contains "1"s for the heads of the undirected edges (*, v).
        B[undirected_heads, undirected_edge_idxs] = 1
        # Incidence matrix contains "1"s for the tails of the undirected edges (v, *).
        B[undirected_tails, undirected_edge_idxs] = 1
    else:
        raise ValueError(f"{matrix_type} is not a valid matrix type!")

    if q == 0.0:
        # Incidence matrix for the Edge Laplacian contains "1"s for the heads of the directed edges (*, v).
        B[directed_heads, directed_edge_idxs] = 1
        # Incidence matrix for the Edge Laplacian contains "-1"s for the tails of the directed edges (v, *).
        if matrix_type == "orientation-equivariant":
            B[directed_tails, directed_edge_idxs] = -1
        elif matrix_type == "orientation-invariant":
            B[directed_tails, directed_edge_idxs] = 1
        else:
            raise ValueError(f"{matrix_type} is not a valid matrix type!")
    else:
        # Incidence matrix for the Magnetic Edge Laplacian contains "e^{-i \pi q}"s for the heads of the directed edges (*, v).
        B[directed_heads, directed_edge_idxs] = np.exp(-1j * np.pi * q)
        # Incidence matrix for the Magnetic Edge Laplacian contains "-e^{i \pi q}"s for the tails of the directed edges (v, *).
        if matrix_type == "orientation-equivariant":
            B[directed_tails, directed_edge_idxs] = -np.exp(1j * np.pi * q)
        elif matrix_type == "orientation-invariant":
            B[directed_tails, directed_edge_idxs] = np.exp(1j * np.pi * q)

    return B


def line_graph_laplacian(edge_index: torch.Tensor):
    """
    Computes the Laplacian matrix of the line graph based on the edge list.

    Args:
        edge_index (torch.Tensor): Edge list.

    Returns:
        torch.Tensor: The Edge Laplacian matrix.
    """
    undirected_mask = torch.ones(
        size=(edge_index.shape[1],),
        device=edge_index.device,
        dtype=bool,
    )

    B = incidence_matrix(
        edge_index=edge_index,
        undirected_mask=undirected_mask,
        matrix_type="orientation-equivariant",
    )
    A = torch.abs(torch.transpose(B, -2, -1) @ B)
    A.fill_diagonal_(0)

    D = torch.diag(A.sum(axis=1))
    L = D - A

    return L


def edge_laplacian(
    edge_index: torch.Tensor,
    undirected_mask: torch.Tensor,
    matrix_type: str,
    return_boundary: bool = False,
):
    """
    Computes the Edge Laplacian matrix based on the edge list.

    Args:
        edge_index (torch.Tensor): Edge list.
        undirected_mask (torch.Tensor): Whether the edges are undirected (label True) or directed (label False).
        matrix_type (str): Whether the matrix is orientation-equivariant or orientation-invariant.

    Returns:
        torch.Tensor: The Edge Laplacian matrix.
    """
    match matrix_type:
        case "orientation-equivariant" | "orientation-invariant":
            B_in = B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type=matrix_type,
            )
        case "mixed-orientation-equivariant-inputs":
            B_in = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-equivariant",
            )
            B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-invariant",
            )
        case "mixed-orientation-invariant-inputs":
            B_in = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-invariant",
            )
            B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-equivariant",
            )
        case _:
            raise ValueError(f"{matrix_type} is not a valid matrix type!")

    L = torch.transpose(B_out, -2, -1) @ B_in
    if return_boundary:
        return L, B_in, B_out
    else:
        return L


def compute_relative_potential(edge_index: torch.Tensor, q: float):
    """
    Computes relative potential based on the topology of the graph.

    Args:
        edge_index (torch.Tensor): Edge list.
        q (float): Absolute potential.

    Returns:
        float: Relative potential.
    """
    q_rel = q / max(edge_index.shape[1], 1)
    return q_rel


def magnetic_edge_laplacian(
    edge_index: torch.Tensor,
    undirected_mask: torch.Tensor,
    matrix_type: str,
    q: float,
    return_boundary: bool = False,
):
    """
    Computes the Magnetic Edge Laplacian matrix based on the edge list.

    Args:
        edge_index (torch.Tensor): Edge list.
        undirected_mask (torch.Tensor): Whether the edges are undirected (label True) or directed (label False).
        matrix_type (str): Whether the matrix is orientation-equivariant or orientation-invariant.
        q (float): Absolute potential.

    Returns:
        torch.Tensor: The Magnetic Edge Laplacian matrix.
    """
    q_rel = compute_relative_potential(edge_index=edge_index, q=q)

    match matrix_type:
        case "orientation-equivariant" | "orientation-invariant":
            B_in = B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type=matrix_type,
                q=q_rel,
            )
        case "mixed-orientation-equivariant-inputs":
            B_in = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-equivariant",
                q=q_rel,
            )
            B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-invariant",
                q=q_rel,
            )
        case "mixed-orientation-invariant-inputs":
            B_in = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-invariant",
                q=q_rel,
            )
            B_out = incidence_matrix(
                edge_index=edge_index,
                undirected_mask=undirected_mask,
                matrix_type="orientation-equivariant",
                q=q_rel,
            )
        case _:
            raise ValueError(f"{matrix_type} is not a valid matrix type!")
    L = torch.conj(torch.transpose(B_out, -2, -1)) @ B_in

    if return_boundary:
        return L, torch.conj(torch.transpose(B_out, -2, -1)), B_in
    else:
        return L


def degree_normalization(matrix: torch.Tensor, return_deg_inv_sqrt: bool = False):
    """
    Normalizes the matrix based on the square roots of the out-degrees like GCN.

    Args:
        matrix (torch.Tensor): Matrix.

    Returns:
        torch.Tensor: Degree normalized matrix.
    """
    deg = torch.abs(matrix).sum(dim=-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    if return_deg_inv_sqrt:
        return deg_inv_sqrt
    else:
        normalized_matrix = (
            deg_inv_sqrt.reshape(-1, 1) * matrix * deg_inv_sqrt.reshape(1, -1)
        )
        return normalized_matrix


if __name__ == "__main__":
    edge_index = torch.Tensor([[0, 1, 2, 2], [2, 2, 3, 4]])
    print(line_graph_laplacian(edge_index=edge_index))

    edge_index = torch.Tensor([[0, 1, 2, 2], [2, 2, 3, 4]])
    undirected_mask = torch.Tensor([0, 0, 0, 0])
    print(
        edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type="orientation-equivariant",
        )
    )

    edge_index = torch.Tensor([[0, 1, 2, 2], [2, 2, 3, 4]])
    undirected_mask = torch.Tensor([0, 0, 0, 0])
    print(
        magnetic_edge_laplacian(
            edge_index=edge_index,
            undirected_mask=undirected_mask,
            matrix_type="orientation-equivariant",
            q=1,
        )
    )  # q_rel = 0.25

    matrix = torch.Tensor([[1, 0, 0], [3, 3, 3], [2, 1, 1]])
    print(degree_normalization(matrix=matrix))
