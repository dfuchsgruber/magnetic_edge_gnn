"""Unit tests for equivariance and invariance properties of the GNNs."""


import itertools
import unittest

import torch

from magnetic_edge_gnn.models.gnn import (
    DirGNN,
    EdgeGNN,
    MagneticChebGNN,
    MagneticEdgeGNN,
    MagneticEdgeGNNHiddenState,
)
from magnetic_edge_gnn.models.gnn.gnn_layers import EdgeConv, MagneticEdgeConv
from magnetic_edge_gnn.models.gnn.magnetic_edge_gnn_hidden import (
    MagneticEdgeGNNHiddenStateBoth,
)
from magnetic_edge_gnn.models.model_utils import (
    degree_normalization,
    incidence_matrix,
    magnetic_edge_laplacian,
)


class TestEquivariances(unittest.TestCase):
    def _random_graph(
        self,
        n=200,
        m_max=1000,
        fraction_undirected: float = 0.5,
        fraction_flips: float = 0.1,
        d: int = 3,
    ):
        edge_idxs = torch.tensor(
            [
                list(edge)
                for edge in set(
                    frozenset(e) for e in torch.randint(0, n, (m_max, 2)).tolist()
                )
                if len(edge) > 1
            ],
            dtype=torch.long,
        ).T
        m = edge_idxs.size(1)
        undirected_mask = torch.rand(m) < fraction_undirected

        edge_attr_equi = torch.randn(m, d)
        edge_attr_inv = torch.randn(m, d)

        return edge_idxs, undirected_mask, n, m, edge_attr_equi, edge_attr_inv

    def test_invariant_node_features(self):
        """Check if the invariance / equivariance properties of incidence matrices hold."""
        fraction_flipped = 0.1
        edge_idxs, undirected_mask, n, m, edge_attr_equi, edge_attr_inv = (
            self._random_graph()
        )
        mask_flipped = (torch.rand(m) < fraction_flipped) & undirected_mask
        edge_idxs_flipped = edge_idxs.clone()
        edge_idxs_flipped[:, mask_flipped] = edge_idxs_flipped[:, mask_flipped].flip(0)
        edge_attr_equi_flipped = edge_attr_equi.clone()
        edge_attr_equi_flipped[mask_flipped] = -edge_attr_equi_flipped[mask_flipped]

        for q in (0.0, 0.1, 0.5, 1.0):
            B_inv = incidence_matrix(
                edge_index=edge_idxs,
                undirected_mask=undirected_mask,
                q=0.0,
                matrix_type="orientation-invariant",
            )
            B_inv_flipped = incidence_matrix(
                edge_index=edge_idxs_flipped,
                undirected_mask=undirected_mask,
                q=0.0,
                matrix_type="orientation-invariant",
            )
            B_equi = incidence_matrix(
                edge_index=edge_idxs,
                undirected_mask=undirected_mask,
                q=0.0,
                matrix_type="orientation-equivariant",
            )
            B_equi_flipped = incidence_matrix(
                edge_index=edge_idxs_flipped,
                undirected_mask=undirected_mask,
                q=0.0,
                matrix_type="orientation-equivariant",
            )

            # Get invariant node features from equivariant features
            x_nodes = B_equi @ edge_attr_equi
            x_nodes2 = B_equi_flipped @ edge_attr_equi_flipped
            assert torch.allclose(x_nodes, x_nodes2, atol=1e-6)

            # Get invariant node features from invariant features
            x_nodes = B_inv @ edge_attr_inv
            x_nodes2 = B_inv_flipped @ edge_attr_inv
            assert torch.allclose(x_nodes, x_nodes2, atol=1e-6)

    def test_mixed_laplacian_equi_and_invariances(self):
        """Tests invariance and equivariance properties of the mixed laplacian."""
        fraction_flipped = 0.1
        edge_idxs, undirected_mask, n, m, edge_attr_equi, edge_attr_inv = (
            self._random_graph()
        )
        mask_flipped = (torch.rand(m) < fraction_flipped) & undirected_mask
        edge_idxs_flipped = edge_idxs.clone()
        edge_idxs_flipped[:, mask_flipped] = edge_idxs_flipped[:, mask_flipped].flip(0)
        edge_attr_equi_flipped = edge_attr_equi.clone()
        edge_attr_equi_flipped[mask_flipped] = -edge_attr_equi_flipped[mask_flipped]

        for q in (0.0, 0.1, 0.5, 1.0):
            L_equi_to_inv = magnetic_edge_laplacian(
                edge_index=edge_idxs,
                undirected_mask=undirected_mask,
                matrix_type="mixed-orientation-equivariant-inputs",
                q=q,
            )
            L_equi_to_inv_flipped = magnetic_edge_laplacian(
                edge_index=edge_idxs_flipped,
                undirected_mask=undirected_mask,
                matrix_type="mixed-orientation-equivariant-inputs",
                q=q,
            )
            L_inv_to_equi = magnetic_edge_laplacian(
                edge_index=edge_idxs,
                undirected_mask=undirected_mask,
                matrix_type="mixed-orientation-invariant-inputs",
                q=q,
            )
            L_inv_to_equi_flipped = magnetic_edge_laplacian(
                edge_index=edge_idxs_flipped,
                undirected_mask=undirected_mask,
                matrix_type="mixed-orientation-invariant-inputs",
                q=q,
            )
            L_equi_to_inv = degree_normalization(L_equi_to_inv)
            L_equi_to_inv_flipped = degree_normalization(L_equi_to_inv_flipped)
            L_inv_to_equi = degree_normalization(L_inv_to_equi)
            L_inv_to_equi_flipped = degree_normalization(L_inv_to_equi_flipped)

            # equivariant signal from invariant inputs
            h_equi = L_inv_to_equi @ edge_attr_inv.to(L_inv_to_equi.dtype)
            h_equi_flipped = L_inv_to_equi_flipped @ edge_attr_inv.to(
                L_inv_to_equi_flipped.dtype
            )
            h_equi_flipped[mask_flipped] = -h_equi_flipped[mask_flipped]
            self.assertTrue(
                torch.allclose(h_equi, h_equi_flipped, atol=1e-6),
                (h_equi, h_equi_flipped),
            )

            # invariant signal from equivariant inputs
            h_inv = L_equi_to_inv @ edge_attr_equi.to(L_equi_to_inv.dtype)
            h_inv_flipped = L_equi_to_inv_flipped @ edge_attr_equi_flipped.to(
                L_equi_to_inv_flipped.dtype
            )
            self.assertTrue(
                torch.allclose(h_inv, h_inv_flipped, atol=1e-6), (h_inv, h_inv_flipped)
            )

    def _test_equi_to_inv_conv(self, conv_cls, complex_signals=False):
        """Tests convolution from equivariant to invariant inputs."""
        fraction_flipped = 0.9
        edge_idxs, undirected_mask, n, m, edge_attr_equi, edge_attr_inv = (
            self._random_graph()
        )
        mask_flipped = (torch.rand(m) < fraction_flipped) & undirected_mask
        edge_idxs_flipped = edge_idxs.clone()
        edge_idxs_flipped[:, mask_flipped] = edge_idxs_flipped[:, mask_flipped].flip(0)
        edge_attr_equi_flipped = edge_attr_equi.clone()
        edge_attr_equi_flipped[mask_flipped] = -edge_attr_equi_flipped[mask_flipped]

        conv_equi_inv = conv_cls(
            in_channels=edge_attr_equi.size(1),
            out_channels=10,
            q=0.5,
            matrix_type="mixed-orientation-equivariant-inputs",
            bias=False,
            skip_connection=False,
        )
        for name, param in conv_equi_inv.named_parameters():
            if "bias" in name:
                param.data.fill_(1.0)

        x_inv = conv_equi_inv(
            edge_idxs,
            edge_attr_equi,
            undirected_mask,
        )
        x_inv_flipped = conv_equi_inv(
            edge_idxs_flipped,
            edge_attr_equi_flipped,
            undirected_mask,
        )
        self.assertTrue(
            torch.allclose(x_inv, x_inv_flipped, atol=1e-6),
            (
                x_inv[(~torch.isclose(x_inv, x_inv_flipped, atol=1e-6)).any(-1)],
                x_inv_flipped[
                    (~torch.isclose(x_inv, x_inv_flipped, atol=1e-6)).any(-1)
                ],
            ),
        )

    def _equi_inv_fusion_kwargs_iterator(self):
        for (
            equivariant_to_invariant,
            invariant_to_equivariant,
            use_fusion,
        ) in itertools.product((False, True), repeat=3):
            if use_fusion:
                equi_hidden_channels = inv_hidden_channels = 8
            else:
                equi_hidden_channels = 8
                inv_hidden_channels = 10
            yield dict(
                invariant_to_equivariant=invariant_to_equivariant,
                equivariant_to_invariant=equivariant_to_invariant,
                equi_hidden_channels=equi_hidden_channels,
                inv_hidden_channels=inv_hidden_channels,
            )

    def test_magnetic_edge_conv_equi_to_inv(self):
        """Tests the equivariant to invariant convolution."""
        self._test_equi_to_inv_conv(MagneticEdgeConv, complex_signals=True)

    def test_edge_conv_equi_to_inv(self):
        """Tests the equivariant to invariant convolution."""
        self._test_equi_to_inv_conv(EdgeConv)

    def _test_model_orientation_equivariant_or_invariant(
        self,
        model,
        fraction_flipped: float = 0.1,
        num_features: int = 7,
        which="equivariant",
    ):
        """Assert the model is equivariant w.r.t. orientation flips of undirected edges."""
        edge_idxs, undirected_mask, n, m, edge_attr_equi, edge_attr_inv = (
            self._random_graph(d=num_features)
        )

        out = model(
            edge_index=edge_idxs,
            equi_edge_attr=edge_attr_equi,
            inv_edge_attr=edge_attr_inv,
            undirected_mask=undirected_mask,
        )

        mask_flipped = (torch.rand(m) < fraction_flipped) & undirected_mask
        # Flip the orientation of some undirected edges
        edge_idxs_flipped = edge_idxs.clone()
        edge_idxs_flipped[:, mask_flipped] = edge_idxs_flipped[:, mask_flipped].flip(0)
        edge_attr_equi_flipped = edge_attr_equi.clone()
        edge_attr_equi_flipped[mask_flipped] = -edge_attr_equi_flipped[mask_flipped]
        out_flipped = model(
            edge_index=edge_idxs_flipped,
            equi_edge_attr=edge_attr_equi_flipped,
            inv_edge_attr=edge_attr_inv,
            undirected_mask=undirected_mask,
        )
        out_flipped = out_flipped.clone()
        match which:
            case "equivariant":
                out_flipped[mask_flipped] = -out_flipped[mask_flipped]
            case "invariant":
                pass
            case _:
                raise ValueError(f"Unknown which: {which}")

        # Assert equivariance
        self.assertTrue(
            torch.allclose(out, out_flipped, atol=1e-6),
            (
                out[~torch.isclose(out, out_flipped, atol=1e-6)],
                out_flipped[~torch.isclose(out, out_flipped, atol=1e-6)],
            ),
        )

    def test_magnetic_edge_gnn_orientation_equivariance(self):
        """Test whether the MagneticEdgeGNN is orientation equivariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=1,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data = torch.randn_like(param.data)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )

    def test_edge_gnn_orientation_equivariance(self):
        """Test whether the EdgeGNN is orientation equivariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = EdgeGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=1,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )

    def test_magnetic_edge_gnn_orientation_invariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=1,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_edge_gnn_orientation_invariance(self):
        """Test whether the EdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = EdgeGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_magnetic_edge_gnn_hidden_orientation_invariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNNHiddenState(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_magnetic_edge_gnn_hidden_orientation_equivariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNNHiddenState(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )

    def test_magnetic_cheb_gnn_invariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticChebGNN(
                degree=5,
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_magnetic_cheb_gnn_hidden_orientation_equivariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticChebGNN(
                degree=5,
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )

    def test_dir_gnn_invariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = DirGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_dir_gnn_equivariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = DirGNN(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )

    def test_magnetic_edge_gnn_hidden_both_orientation_invariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNNHiddenStateBoth(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=False,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="invariant"
            )

    def test_magnetic_edge_gnn_hidden_both_orientation_equivariance(self):
        """Test whether the MagneticEdgeGNN is orientation invariant."""
        d_in = 7
        for kwargs in self._equi_inv_fusion_kwargs_iterator():
            model = MagneticEdgeGNNHiddenStateBoth(
                equi_in_channels=d_in,
                inv_in_channels=d_in,
                out_channels=12,
                orientation_equivariant_labels=True,
                num_layers=4,
                dropout=0.0,
                equi_act="tanh",
                inv_act="relu",
                q=0.35,
                **kwargs,
            )
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.data.fill_(1.0)
            self._test_model_orientation_equivariant_or_invariant(
                model, num_features=d_in, fraction_flipped=0.5, which="equivariant"
            )
