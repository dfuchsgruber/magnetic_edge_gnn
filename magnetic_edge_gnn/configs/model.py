from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def mlp():
    model = dict(
        type_="MLP",
        name="mlp",
    )


@experiment.named_config
def mlp_with_fusion():
    model = dict(
        type_="MLPWithFusion",
        name="mlp_with_fusion",
    )


@experiment.named_config
def line_graph_gnn():
    model = dict(
        type_="LineGraphGNN",
        name="line_graph_gnn",
    )


@experiment.named_config
def hodge_gnn():
    model = dict(
        type_="HodgeGNN",
        name="hodge_gnn",
    )
    data = dict(
        arbitrary_orientation=True,
    )


@experiment.named_config
def equivariant_hodge_gnn():
    model = dict(
        type_="HodgeGNN",
        name="equivariant_hodge_gnn",
        inputs="equivariant",
    )
    data = dict(
        arbitrary_orientation=True,
    )


@experiment.named_config
def concatenated_hodge_gnn():
    model = dict(
        type_="HodgeGNN",
        name="concatenated_hodge_gnn",
        inputs="both",
    )
    data = dict(
        arbitrary_orientation=True,
    )


@experiment.named_config
def directed_hodge_gnn():
    model = dict(
        type_="DirectedHodgeGNN",
        name="directed_hodge_gnn",
    )


@experiment.named_config
def edge_gnn():
    model = dict(
        type_="EdgeGNN",
        name="edge_gnn",
    )


@experiment.named_config
def magnetic_edge_gnn():
    model = dict(
        type_="MagneticEdgeGNN",
        name="magnetic_edge_gnn",
    )


@experiment.named_config
def magnetic_edge_gnn_no_fusion():
    model = dict(
        type_="MagneticEdgeGNN",
        name="magnetic_edge_gnn_no_fusion",
        use_fusion_layers=False,
    )


@experiment.named_config
def mixed_magnetic_edge_gnn_and_fusion():
    model = dict(
        type_="MagneticEdgeGNN",
        name="mixed_magnetic_edge_gnn_and_fusion",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
    )


@experiment.named_config
def mixed_edge_gnn_and_fusion():
    model = dict(
        type_="EdgeGNN",
        name="mixed_edge_gnn_and_fusion",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
    )


@experiment.named_config
def mixed_magnetic_edge_gnn():
    model = dict(
        type_="MagneticEdgeGNN",
        name="mixed_magnetic_edge_gnn",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
        use_fusion_layers=False,
    )


@experiment.named_config
def mixed_magnetic_edge_gnn_hidden():
    model = dict(
        type_="MagneticEdgeGNNHidden",
        name="mixed_magnetic_edge_gnn_hidden_node_potentials",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
        use_fusion_layers=False,
    )


@experiment.named_config
def mixed_magnetic_edge_gnn_hidden_and_fusion():
    model = dict(
        type_="MagneticEdgeGNNHidden",
        name="mixed_magnetic_edge_gnn_hidden_node_potentials_and_fusion",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
        use_fusion_layers=True,
    )


@experiment.named_config
def mixed_magnetic_edge_gnn_hidden_both_and_fusion():
    model = dict(
        type_="MagneticEdgeGNNHiddenBoth",
        name="mixed_magnetic_edge_gnn_hidden_node_potentials_both_and_fusion",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
        use_fusion_layers=True,
    )


@experiment.named_config
def eign():
    model = dict(
        type_="MagneticEdgeGNNHidden",
        name="mixed_magnetic_edge_gnn_hidden_node_potentials_and_fusion",
        invariant_to_equivariant=True,
        equivariant_to_invariant=True,
        use_fusion_layers=True,
    )

@experiment.named_config
def transformer():
    data = dict(
        laplacian_encodings_phase_shift=1.0,
        num_laplacian_encodings=32,
    )
    model = dict(
        type_="Transformer",
        name="transformer",
        num_heads=8,
        num_pos_encodings=32,
    )


@experiment.named_config
def dir_gnn():
    model = dict(
        type_="DirGNN",
        name="dir_gnn",
    )
