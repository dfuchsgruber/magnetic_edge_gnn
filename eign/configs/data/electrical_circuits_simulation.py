from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def electrical_circuits_simulation():
    save_dir = "output/runs/eletrical-circuits-simulation"
    data = dict(
        name="electrical-circuits-simulation",
        dataset_path="data/electrical-circuits/v1",
        orientation_equivariant_labels=True,
        val_ratio=0.25,
        test_ratio=0.25,
        include_non_source_voltages=False,
        current_relative_to_voltage=True,
    )
    model = dict(
        equi_input_dim=2,
        inv_input_dim=4,
        hidden_dim=32,
        output_dim=1,
        num_layers=4,
        dropout=0.1,
        q=1.0,
    )
    training = dict(
        num_epochs=200,
        batch_size=10,
        loss="mse_loss",
        max_grad_norm=1.0,
    )
    optimization = dict(
        optim="adam",
        lr=3.0e-3,
        weight_decay=0.0,
    )
