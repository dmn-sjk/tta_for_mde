import os
import importlib
import hashlib

import numpy as np
import torch


class ExperimentLogger:
    """Backend-agnostic experiment logger with a writer-compatible API."""

    def __init__(self, log_dir, backend="wandb", run_name=None, project="ada-depth",
                 entity=None, group=None, config=None):
        self.log_dir = log_dir
        self.backend = backend
        self.run = None
        self.writer = None

        os.makedirs(self.log_dir, exist_ok=True)

        if self.backend == "tensorboard":
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        elif self.backend == "wandb":
            try:
                wandb = importlib.import_module("wandb")
            except ImportError as exc:
                raise ImportError(
                    "wandb is the default logging backend. Install wandb or run with "
                    "--logging_backend tensorboard."
                ) from exc

            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=self._limit_wandb_name(run_name),
                group=self._limit_wandb_name(group),
                dir=self.log_dir,
                config=self._sanitize_config(config),
                reinit=True,
            )
        else:
            raise ValueError("Unknown logging backend: {}".format(self.backend))

    @classmethod
    def from_options(cls, opt, log_dir, run_name=None):
        return cls(
            log_dir=log_dir,
            backend=opt.logging_backend,
            run_name=run_name or opt.model_name,
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            group=opt.model_name,
            config=vars(opt),
        )

    def add_scalar(self, tag, scalar_value, step):
        if self.backend == "tensorboard":
            self.writer.add_scalar(tag, scalar_value, step)
            return

        self.run.log({tag: self._to_scalar(scalar_value)}, step=step)

    def add_image(self, tag, img_tensor, step):
        if self.backend == "tensorboard":
            self.writer.add_image(tag, img_tensor, step)
            return

        self.run.log({tag: self.wandb.Image(self._to_image(img_tensor))}, step=step)

    def add_figure(self, tag, figure, step):
        if self.backend == "tensorboard":
            self.writer.add_figure(tag, figure, step)
            return

        self.run.log({tag: self.wandb.Image(figure)}, step=step)

    def add_summary(self, summary, prefix, step):
        flat_summary = self._flatten_scalars(summary, prefix)
        if self.backend == "tensorboard":
            for tag, value in flat_summary.items():
                self.writer.add_scalar(tag, value, step)
            return

        for tag, value in flat_summary.items():
            self.run.summary[tag] = value
        if flat_summary:
            self.run.log(flat_summary, step=step)

    def log_config(self, config):
        if self.backend != "wandb":
            return

        if not isinstance(config, dict):
            config = vars(config)
        self.run.config.update(self._sanitize_config(config), allow_val_change=True)

    def flush(self):
        if self.backend == "tensorboard":
            self.writer.flush()

    def close(self):
        if self.backend == "tensorboard":
            self.writer.close()
        elif self.run is not None:
            self.run.finish()

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _to_image(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        return image

    @staticmethod
    def _limit_wandb_name(value, max_length=128):
        if value is None:
            return None
        value = str(value)
        if len(value) <= max_length:
            return value

        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        keep = max_length - len(digest) - 1
        return "{}-{}".format(value[:keep], digest)

    @classmethod
    def _flatten_scalars(cls, values, prefix):
        flattened = {}
        for key, value in values.items():
            tag = "{}/{}".format(prefix, key) if prefix else key
            if isinstance(value, dict):
                flattened.update(cls._flatten_scalars(value, tag))
                continue

            scalar_value = cls._to_scalar(value)
            if isinstance(scalar_value, (int, float)):
                flattened[tag] = scalar_value
        return flattened

    @classmethod
    def _sanitize_config(cls, config):
        if config is None:
            return None
        if isinstance(config, dict):
            return {key: cls._sanitize_config(value) for key, value in config.items()}
        if isinstance(config, (list, tuple)):
            return [cls._sanitize_config(value) for value in config]
        if isinstance(config, (str, int, float, bool)) or config is None:
            return config
        return str(config)
