from magnetic_edge_gnn.experiment import experiment


def get_traffic_flow_interpolation_default_config():
    return dict(
        save_dir="output/traffic-flow-interpolation",
        checkpoint=True,
        data=dict(
            orientation_equivariant_labels=True,
        val_ratio=0.1,
        test_ratio=0.1,
        ),
        model=dict(
            hidden_dim=32,
            output_dim=1,
            num_layers=4,
            dropout=0.1,
            q=1.0,
        ),
        training=dict(num_epochs=500, batch_size=1, loss="mse_loss", max_grad_norm=1.0),
        optimization=dict(optim="adam", lr=3.0e-3, weight_decay=0.0),
    )


@experiment.named_config
def traffic_flow_interpolation_anaheim():
    locals().update(get_traffic_flow_interpolation_default_config())
    save_dir = "output/runs/traffic-flow-interpolation-anaheim"
    data |= dict(
        name="traffic-anaheim-interpolation",
        dataset_path="data/traffic-anaheim",
    )
    model |= dict(
        equi_input_dim=1,
        inv_input_dim=7 + 1,
    )


@experiment.named_config
def traffic_flow_interpolation_barcelona():
    locals().update(get_traffic_flow_interpolation_default_config())
    save_dir = "output/runs/traffic-flow-interpolation-barcelona"
    data |= dict(
        name="traffic-barcelona-interpolation",
        dataset_path="data/traffic-barcelona",
    )
    model |= dict(
        equi_input_dim=1,
        inv_input_dim=8 + 1,
    )


@experiment.named_config
def traffic_flow_interpolation_chicago():
    locals().update(get_traffic_flow_interpolation_default_config())
    save_dir = "output/runs/traffic-flow-interpolation-chicago"
    data |= dict(
        name="traffic-chicago-interpolation",
        dataset_path="data/traffic-chicago",
    )
    model |= dict(
        equi_input_dim=1,
        inv_input_dim=9 + 1,
    )


@experiment.named_config
def traffic_flow_interpolation_winnipeg():
    locals().update(get_traffic_flow_interpolation_default_config())
    save_dir = "output/runs/traffic-flow-interpolation-winnipeg"
    data |= dict(
        name="traffic-winnipeg-interpolation",
        dataset_path="data/traffic-winnipeg",
    )
    model |= dict(
        equi_input_dim=1,
        inv_input_dim=7 + 1,
    )
