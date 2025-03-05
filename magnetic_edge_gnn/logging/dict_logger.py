from typing import Iterable

import plotext as plt
from pytorch_lightning.loggers import Logger


class DictLogger(Logger):
    """Logger that stores metrics in a dictionary in memory."""
    def __init__(self):
        super().__init__()
        self.metrics = []  # List to store all metrics

    @property
    def experiment(self):
        return None

    @property
    def name(self):
        return "DictLogger"

    @property
    def version(self):
        return "0"

    def log_metrics(self, metrics, step):
        # Append metrics for each step (e.g., epoch end)
        self.metrics.append(metrics)

    def log_hyperparams(self, params):
        pass

    def save(self):
        pass

    def finalize(self, status):
        pass

    def get_metrics(self) -> dict[str, list]:
        # Convert metrics list to a dictionary of lists
        dict_of_lists = {}
        for metric_dict in self.metrics:
            for key, value in metric_dict.items():
                if key not in dict_of_lists:
                    dict_of_lists[key] = []
                dict_of_lists[key].append(value)
        return dict_of_lists

    def print_metrics(self, *metrics: Iterable[str]):
        plt.clear_figure()
        values = self.get_metrics()
        for metric in metrics:
            assert metric in values, f"Metric {metric} not found."

            plt.plot(range(len(values[metric])), values[metric], label=metric)
        plt.show()
