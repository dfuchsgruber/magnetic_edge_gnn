from magnetic_edge_gnn.experiment import experiment

default_config = dict(
    # Seml stuff
    db_collection=None,
    overwrite=None,
    save_dir="./output",
    checkpoint=True,
    data=dict(
        val_ratio=0.1,
        test_ratio=0.2,
        arbitrary_orientation=True,
        interpolation_label_size=0.1,
        registry=dict(
            database_path="./data/registry/registry.db",
            lockfile_path="./data/registry/registry.lock",
            storage_path="./data/registry/storage",
            force_rebuild=False,
        ),
        laplacian_encodings_phase_shift=0.0,
        num_laplacian_encodings=0,
    ),
    model=dict(
        equi_input_dim=-1,  # num equivariant inputs
        inv_input_dim=-1,  # num invariant inputs
        hidden_dim=32,  # hidden dimension
        output_dim=1,  # output dimension
        num_layers=2,  # number of layers
        dropout=0.0,  # dropout rate
        q=1.0,  # phase shift parameter for the magnetic laplacian
        equivariant_to_invariant=False,
        invariant_to_equivariant=False,
        use_fusion_layers=True,
        inputs="both",
        num_pos_encodings=0,
    ),
    training=dict(
        num_epochs=50,
        batch_size=10,
        loss="bce_loss",
        max_grad_norm=1.0,
    ),
    optimization=dict(
        optim="adam",
        lr=1e-3,
        weight_decay=0.0,
    ),
    logging=dict(
        silent=None,  # Determined by db_collection and overwrite
        plot_to_console=(("train/loss", "val/loss"), ("val/rmse", "test/rmse")),
        wandb=None,
    ),
    split_idx=None,
    run_idx=None,
)


@experiment.config
def default():
    locals().update(default_config)
