from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def typed_triangles_orientation():
    save_dir = "output/runs/typed-triangles-orientation"
    checkpoint = True
    data = dict(
        name="typed-triangles-orientation",
        dataset_path="data/typed-triangles-orientation",
        orientation_equivariant_labels=True,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    model = dict(
        equi_input_dim=2,
        inv_input_dim=3,
        hidden_dim=32,
        output_dim=1,
        num_layers=4,
        dropout=0.0,
        q=1.0,
    )
    training = dict(num_epochs=50, batch_size=1, loss="mse_loss", max_grad_norm=1.0)
    optimization = dict(optim="adam", lr=1.0e-2, weight_decay=0.0)
