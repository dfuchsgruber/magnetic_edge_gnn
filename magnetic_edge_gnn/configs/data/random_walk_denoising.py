from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def random_walk_denoising():
    save_dir = "output/runs/random-walk-denoising"
    checkpoint = True
    data = dict(
        name="random-walk-denoising",
        dataset_path="data/random-walk-denoising",
        orientation_equivariant_labels=False,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    model = dict(
        equi_input_dim=1,
        inv_input_dim=2,
        hidden_dim=32,
        output_dim=1,
        num_layers=4,
        dropout=0.1,
        q=1.0,
    )
    training = dict(num_epochs=50, batch_size=10, loss="bce_loss", max_grad_norm=1.0)
    optimization = dict(optim="adam", lr=3.0e-3, weight_decay=0.0)
    logging = dict(
        plot_to_console=(
            ("train/loss", "val/loss"),
            ("val/auc_roc", "test/auc_roc"),
        ),
    )
