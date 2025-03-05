from magnetic_edge_gnn.experiment import experiment


@experiment.named_config
def wandb(model, optimization, data):
    logging = dict(
        wandb=dict(
            project=f"mel-{data['name']}",
            name=f"{model['name']}-layers_{model['num_layers']}-dim_{model['hidden_dim']}-lr_{optimization['lr']}",
            log_model=False,
            offline=False,
            # These are custom fields not directly passed to wandb.init
            log_internal_dir=str(
                str("output/wandb")
            ),  # this is a dummy directory that links to /dev/null to not save the internal logs
            cache_dir=str("output/wandb_cache"),
            api_token_path="./wandb_api_token",
        )
    )
