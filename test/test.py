"""Unit tests for equivariance and invariance properties of the GNNs."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from eign.block import (
    EIGNBlockMagneticEdgeLaplacianConv,
    EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv,
)
from eign.conv import (
    MagneticEdgeLaplacianConv,
    MagneticEdgeLaplacianWithNodeTransformationConv,
)
from eign.eign import EIGNLaplacianConv, EIGNLaplacianWithNodeTransformationConv
from eign.laplacian import magnetic_edge_laplacian


@dataclass
class Graph:
    edge_idxs: torch.Tensor
    is_directed: torch.Tensor
    num_nodes: int
    num_edges: int
    signed_edge_attr: torch.Tensor
    unsigned_edge_attr: torch.Tensor

    def __repr__(self):
        return (
            f'Graph(n={self.num_nodes}, m={self.num_edges}, '
            f'fraction_undirected={self.is_directed.float().mean()}, '
            f'd={self.signed_edge_attr.size(1)})'
        )


def random_graph(
    n=200,
    m_max=1000,
    fraction_undirected: float = 0.5,
    d: int = 3,
) -> Graph:
    edge_idxs = torch.tensor(
        [
            list(edge)
            for edge in {frozenset(e) for e in torch.randint(0, n, (m_max, 2)).tolist()}
            if len(edge) > 1
        ],
        dtype=torch.long,
    ).T
    m = edge_idxs.size(1)
    is_directed = torch.rand(m) >= fraction_undirected

    edge_attr_equi = torch.randn(m, d)
    edge_attr_inv = torch.randn(m, d)

    return Graph(
        edge_idxs=edge_idxs,
        is_directed=is_directed,
        num_nodes=n,
        num_edges=m,
        signed_edge_attr=edge_attr_equi,
        unsigned_edge_attr=edge_attr_inv,
    )


def reorient_graph(graph, p_reorient: float = 0.1) -> tuple[Graph, torch.Tensor]:
    mask_flipped = (torch.rand(graph.num_edges) < p_reorient) & (~graph.is_directed)
    edge_idxs_flipped = graph.edge_idxs.clone()
    edge_idxs_flipped[:, mask_flipped] = edge_idxs_flipped[:, mask_flipped].flip(0)
    edge_attr_flipped = graph.signed_edge_attr.clone()
    edge_attr_flipped[mask_flipped] = -edge_attr_flipped[mask_flipped]
    return (
        Graph(
            edge_idxs=edge_idxs_flipped,
            is_directed=graph.is_directed,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            signed_edge_attr=edge_attr_flipped,
            unsigned_edge_attr=graph.unsigned_edge_attr,
        ),
        mask_flipped,
    )


test_graphs = [
    random_graph(n=200, m_max=1000, fraction_undirected=0.5, d=3),
    random_graph(n=100, m_max=1000, fraction_undirected=1.0, d=10),
    random_graph(n=200, m_max=1000, fraction_undirected=0.0, d=5),
]


def pytest_generate_tests(metafunc):
    if 'graph' in metafunc.fixturenames:
        metafunc.parametrize('graph', test_graphs)
    if 'use_fusion' in metafunc.fixturenames:
        metafunc.parametrize('use_fusion', [False, True])
    if 'use_signed_to_unsigned_conv' in metafunc.fixturenames:
        metafunc.parametrize('use_signed_to_unsigned_conv', [False, True])
    if 'use_unsigned_to_signed_conv' in metafunc.fixturenames:
        metafunc.parametrize('use_unsigned_to_signed_conv', [False, True])
    if 'signed_in' in metafunc.fixturenames:
        metafunc.parametrize('signed_in', [False, True])
    if 'signed_out' in metafunc.fixturenames:
        metafunc.parametrize('signed_out', [False, True])
    if 'q' in metafunc.fixturenames:
        metafunc.parametrize('q', [0.0, 0.35, 1.0])
    if 'normalize' in metafunc.fixturenames:
        metafunc.parametrize('normalize', [False, True])
    if 'use_residual' in metafunc.fixturenames:
        metafunc.parametrize('use_residual', [False, True])


def _test_function_equivariance_and_invariance(
    f: Callable[[Graph], tuple[torch.Tensor, torch.Tensor]],
    graph: Graph,
    p_reorient: float = 0.1,
):
    """Tests if a function is equivariant and invariant to orientation for undirected edges."""
    out_signed, out_unsigned = f(graph)
    graph_flipped, mask_flipped = reorient_graph(graph, p_reorient=p_reorient)
    out_signed_flipped, out_unsigned_flipped = f(graph_flipped)
    out_signed_flipped = out_signed_flipped.clone()
    out_signed_flipped[mask_flipped] = -out_signed_flipped[mask_flipped]

    # Assert equivariance
    assert torch.allclose(
        out_signed, out_signed_flipped, atol=1e-4
    ), f'No equivariant mapping, differences: {torch.abs(out_signed - out_signed_flipped).max():.3f}'
    # Assert invariance
    assert torch.allclose(
        out_unsigned, out_unsigned_flipped, atol=1e-4
    ), f'No invariant mapping, differences: {torch.abs(out_unsigned - out_unsigned_flipped).max():.3f}'


def test_laplacian_equivariance_and_invariance(
    graph: Graph, signed_in: bool, signed_out: bool, q: float
):
    def f(g):
        if signed_in:
            edge_attr = g.signed_edge_attr
        else:
            edge_attr = g.unsigned_edge_attr
        laplacian = magnetic_edge_laplacian(
            g.edge_idxs,
            g.is_directed,
            signed_in=signed_in,
            signed_out=signed_out,
            q=q,
            return_incidence=False,
        )
        # Change the dtype of edge_attr to match the laplacian (in case of complex numbers)
        edge_attr = edge_attr.to(laplacian.dtype)
        out = laplacian @ edge_attr
        if signed_out:
            return out, g.unsigned_edge_attr
        else:
            return g.signed_edge_attr, out

    _test_function_equivariance_and_invariance(f, graph)


def test_laplacian_edge_conv_equivariance_and_invariance(
    graph: Graph, signed_in: bool, signed_out: bool, q: float, normalize: bool
):
    in_channels = (
        graph.signed_edge_attr.size(1)
        if signed_in
        else graph.unsigned_edge_attr.size(1)
    )
    model = MagneticEdgeLaplacianConv(
        in_channels=in_channels,
        out_channels=2 * in_channels,
        signed_in=signed_in,
        signed_out=signed_out,
        q=q,
        normalize=normalize,
        cached=False,
    ).eval()

    @torch.no_grad()
    def f(g):
        if signed_in:
            edge_attr = g.signed_edge_attr
        else:
            edge_attr = g.unsigned_edge_attr
        out = model(edge_index=g.edge_idxs, x=edge_attr, is_directed=g.is_directed)

        if signed_out:
            return out, g.unsigned_edge_attr
        else:
            return g.signed_edge_attr, out

    _test_function_equivariance_and_invariance(f, graph)


def test_laplacian_edge_conv_with_node_transformation_equivariance_and_invariance(
    graph: Graph, signed_in: bool, signed_out: bool, q: float, normalize: bool
):
    in_channels = (
        graph.signed_edge_attr.size(1)
        if signed_in
        else graph.unsigned_edge_attr.size(1)
    )
    model = MagneticEdgeLaplacianWithNodeTransformationConv(
        in_channels=in_channels,
        out_channels=2 * in_channels,
        initialize_node_feature_transformation=lambda in_channels,
        out_channels: nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        ),
        signed_in=signed_in,
        signed_out=signed_out,
        q=q,
        normalize=normalize,
        cached=False,
    ).eval()

    @torch.no_grad()
    def f(g):
        if signed_in:
            edge_attr = g.signed_edge_attr
        else:
            edge_attr = g.unsigned_edge_attr
        out = model(edge_index=g.edge_idxs, x=edge_attr, is_directed=g.is_directed)

        if signed_out:
            return out, g.unsigned_edge_attr
        else:
            return g.signed_edge_attr, out

    _test_function_equivariance_and_invariance(f, graph)


def test_eign_block_magnetic_edge_laplacian_conv_equivariance_and_invariance(
    graph: Graph,
    q: float,
    normalize: bool,
    use_fusion: bool,
    use_residual: bool,
    use_unsigned_to_signed_conv: bool,
    use_signed_to_unsigned_conv: bool,
):
    block = EIGNBlockMagneticEdgeLaplacianConv(
        in_channels_signed=graph.signed_edge_attr.size(1),
        in_channels_unsigned=graph.unsigned_edge_attr.size(1),
        out_channels_signed=2 * graph.signed_edge_attr.size(1),
        out_channels_unsigned=2 * graph.unsigned_edge_attr.size(1),
        use_fusion=use_fusion,
        use_residual=use_residual,
        use_unsigned_to_signed_conv=use_unsigned_to_signed_conv,
        use_signed_to_unsigned_conv=use_signed_to_unsigned_conv,
        q=q,
        normalize=normalize,
    ).eval()

    @torch.no_grad()
    def f(g):
        out_signed, out_unsigned = block(
            edge_index=g.edge_idxs,
            x_signed=g.signed_edge_attr,
            x_unsigned=g.unsigned_edge_attr,
            is_directed=g.is_directed,
        )
        return out_signed, out_unsigned

    _test_function_equivariance_and_invariance(f, graph)


def test_eign_block_magnetic_edge_laplacian_with_node_transformation_conv_equivariance_and_invariance(
    graph: Graph,
    q: float,
    normalize: bool,
    use_fusion: bool,
    use_residual: bool,
    use_unsigned_to_signed_conv: bool,
    use_signed_to_unsigned_conv: bool,
):
    block = EIGNBlockMagneticEdgeLaplacianWithNodeTransformationConv(
        in_channels_signed=graph.signed_edge_attr.size(1),
        in_channels_unsigned=graph.unsigned_edge_attr.size(1),
        out_channels_signed=2 * graph.signed_edge_attr.size(1),
        out_channels_unsigned=2 * graph.unsigned_edge_attr.size(1),
        use_fusion=use_fusion,
        use_residual=use_residual,
        use_unsigned_to_signed_conv=use_unsigned_to_signed_conv,
        use_signed_to_unsigned_conv=use_signed_to_unsigned_conv,
        q=q,
        normalize=normalize,
        initialize_node_feature_transformation=lambda in_channels,
        out_channels: nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        ),
    ).eval()

    @torch.no_grad()
    def f(g):
        out_signed, out_unsigned = block(
            edge_index=g.edge_idxs,
            x_signed=g.signed_edge_attr,
            x_unsigned=g.unsigned_edge_attr,
            is_directed=g.is_directed,
        )
        return out_signed, out_unsigned

    _test_function_equivariance_and_invariance(f, graph)


def test_eign_laplacian_conv_equivariance_and_invariance(
    graph: Graph,
    q: float,
    normalize: bool,
    use_fusion: bool,
    use_residual: bool,
    use_unsigned_to_signed_conv: bool,
    use_signed_to_unsigned_conv: bool,
):
    model = EIGNLaplacianConv(
        in_channels_signed=graph.signed_edge_attr.size(1),
        in_channels_unsigned=graph.unsigned_edge_attr.size(1),
        out_channels_signed=3,
        out_channels_unsigned=5,
        hidden_channels_signed=2 * graph.signed_edge_attr.size(1),
        hidden_channels_unsigned=2 * graph.unsigned_edge_attr.size(1),
        num_blocks=2,
        use_fusion=use_fusion,
        use_residual=use_residual,
        use_unsigned_to_signed_conv=use_unsigned_to_signed_conv,
        use_signed_to_unsigned_conv=use_signed_to_unsigned_conv,
        q=q,
        normalize=normalize,
    ).eval()

    @torch.no_grad()
    def f(g):
        out_signed, out_unsigned = model(
            edge_index=g.edge_idxs,
            x_signed=g.signed_edge_attr,
            x_unsigned=g.unsigned_edge_attr,
            is_directed=g.is_directed,
        )
        return out_signed, out_unsigned

    _test_function_equivariance_and_invariance(f, graph)


def test_eign_laplacian_with_node_transformation_conv_equivariance_and_invariance(
    graph: Graph,
    q: float,
    normalize: bool,
    use_fusion: bool,
    use_residual: bool,
    use_unsigned_to_signed_conv: bool,
    use_signed_to_unsigned_conv: bool,
):
    model = EIGNLaplacianWithNodeTransformationConv(
        in_channels_signed=graph.signed_edge_attr.size(1),
        in_channels_unsigned=graph.unsigned_edge_attr.size(1),
        out_channels_signed=3,
        out_channels_unsigned=5,
        hidden_channels_signed=2 * graph.signed_edge_attr.size(1),
        hidden_channels_unsigned=2 * graph.unsigned_edge_attr.size(1),
        num_blocks=2,
        use_fusion=use_fusion,
        use_residual=use_residual,
        use_unsigned_to_signed_conv=use_unsigned_to_signed_conv,
        use_signed_to_unsigned_conv=use_signed_to_unsigned_conv,
        q=q,
        normalize=normalize,
    ).eval()

    @torch.no_grad()
    def f(g):
        out_signed, out_unsigned = model(
            edge_index=g.edge_idxs,
            x_signed=g.signed_edge_attr,
            x_unsigned=g.unsigned_edge_attr,
            is_directed=g.is_directed,
        )
        return out_signed, out_unsigned

    _test_function_equivariance_and_invariance(f, graph)
