"""A collection of metric logging classes"""

import abc
import os
import logging

logging.basicConfig(level=logging.INFO)

from datetime import datetime


class MetricLogger(abc.ABC):
    """Abstract class for metric logging"""

    @abc.abstractmethod
    def log_params(self, params: dict):
        """Logs the parameters"""

    @abc.abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """Logs the metrics"""

    @abc.abstractmethod
    def close(self):
        """Closes the logger"""


class StdOutMetricLogger(MetricLogger):
    """Logs the metrics to stdout"""

    def __init__(self, run_name: str, device_id: int):
        self.run_name = run_name
        self.device_id = device_id
        if self.device_id == 0:
            self.setup()

    def setup(self):
        self.logger = logging.getLogger(__name__)
        # set custom formatter with run_name
        formatter = logging.Formatter(
            f"{self.run_name} - {self.device_id} - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_params(self, params: dict):
        """Logs the parameters to stdout"""
        if self.device_id == 0:
            self.logger.info(params)

    def log_metrics(self, metric: dict, step: int):
        """Logs the metrics to stdout"""
        if self.device_id == 0:
            self.logger.info(f"Step: {step}, Metric: {metric}")

    def close(self):
        """Logs the termination message to stdout"""
        if self.device_id == 0:
            self.logger.info("Terminated!")


class MLFlowMetricLogger(MetricLogger):
    """Logs the metrics to MLFlow"""

    def __init__(self, run_name: str, device_id: int):
        self.run_name = run_name
        self.device_id = device_id
        if self.device_id == 0:
            self.setup()

    def setup(self):
        import mlflow

        service_url = os.getenv("MLFLOW_SERVICE_URL", "http://localhost:5000")
        mlflow.set_tracking_uri(service_url)
        self.mlflow_client = mlflow.client.MlflowClient()
        experiment_name = f"cot-icl-lab-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        experiment = mlflow.set_experiment(experiment_name)
        self.run = self.mlflow_client.create_run(
            experiment_id=experiment.experiment_id, run_name=self.run_name
        )

    def log_params(self, params: dict):
        """Logs the parameters to MLFlow"""
        if self.device_id == 0:
            for key, value in params.items():
                self.mlflow_client.log_param(
                    run_id=self.run.info.run_id, key=key, value=value
                )

    def log_metrics(self, metrics: dict, step: int):
        """Logs the metrics to MLFlow"""
        if self.device_id == 0:
            for key, value in metrics.items():
                self.mlflow_client.log_metric(
                    run_id=self.run.info.run_id,
                    key=key,
                    value=value,
                    step=step,
                )

    def close(self):
        """Logs the termination message to MLFlow"""
        if self.device_id == 0:
            self.mlflow_client.set_terminated(self.run.info.run_id)


METRIC_LOGGER_REGISTRY = {
    "stdout": StdOutMetricLogger,
    "mlflow": MLFlowMetricLogger,
}
