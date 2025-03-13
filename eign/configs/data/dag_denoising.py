from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def DAG_denoising():
    save_dir = "output/runs/DAG-denoising"
    checkpoint = True
    data = dict(
        name="DAG-denoising",
        dataset_path="data/DAG-denoising",
        orientation_equivariant_labels=False,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    model = dict(
        equi_input_dim=0,
        inv_input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_layers=4,
        dropout=0.1,
        q=1.0,
    )
    training = dict(num_epochs=30, batch_size=10, loss="bce_loss", max_grad_norm=1.0)
    optimization = dict(optim="adam", lr=3.0e-3, weight_decay=0.0)
