from magnetic_edge_gnn.experiment import experiment


def get_traffic_flow_simulation_default_config():
    return dict(
        save_dir="output/traffic-flow-simulation",
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
def traffic_flow_simulation_anaheim():
    locals().update(get_traffic_flow_simulation_default_config())
    save_dir = "output/runs/traffic-flow-simulation-anaheim"
    data |= dict(
        name="traffic-anaheim-simulation",
        dataset_path="data/traffic-anaheim",
    )
    model |= dict(
        equi_input_dim=1,  # zeros
        inv_input_dim=7 + 1,
    )


@experiment.named_config
def traffic_flow_simulation_barcelona():
    locals().update(get_traffic_flow_simulation_default_config())
    save_dir = "output/runs/traffic-flow-simulation-barcelona"
    data |= dict(
        name="traffic-barcelona-simulation",
        dataset_path="data/traffic-barcelona",
    )
    model |= dict(
        equi_input_dim=1,  # zeros
        inv_input_dim=8 + 1,
    )


@experiment.named_config
def traffic_flow_simulation_chicago():
    locals().update(get_traffic_flow_simulation_default_config())
    save_dir = "output/runs/traffic-flow-simulation-chicago"
    data |= dict(
        name="traffic-chicago-simulation",
        dataset_path="data/traffic-chicago",
    )
    model |= dict(
        equi_input_dim=1,  # zeros
        inv_input_dim=9 + 1,
    )


@experiment.named_config
def traffic_flow_simulation_winnipeg():
    locals().update(get_traffic_flow_simulation_default_config())
    save_dir = "output/runs/traffic-flow-simulation-winnipeg"
    data |= dict(
        name="traffic-winnipeg-simulation",
        dataset_path="data/traffic-winnipeg",
    )
    model |= dict(
        equi_input_dim=1,  # zeros
        inv_input_dim=7 + 1,
    )
