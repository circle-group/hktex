from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import hktex

if TYPE_CHECKING:
    # This import does NOT run at runtime to avoid circular dependencies, but still
    # provides formal type hints for the MixIn class below.
    from .base import MeshSamplerDataModule


@dataclass
class ImportanceSamplingDataConfigMixIn:
    importance_sampling_pool_size: int = 1_000_000
    importance_sampling_warmup_steps: int = 1000
    importance_sampling_ema_beta: float = 0.9


class ImportanceSamplerDataset(IterableDataset):
    """
    A dataset wrapper that builds a large sample pool from an
    "inner" dataset and then provides importance-sampled batches.
    """

    def __init__(
        self, inner_dataset: IterableDataset, cfg: ImportanceSamplingDataConfigMixIn
    ):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.cfg = cfg

        self.pool = None
        self.errors = None
        self.pool_size = 0
        self.step = 0
        self.current_sampling_method = "uniform"

        self._build_pool()
        hktex.debug(f"Pool created with {self.pool_size} samples.")

    def _build_pool(self):
        """
        Automatically builds the pool by iterating over the
        inner dataset.
        """
        inner_iterator = iter(self.inner_dataset)
        pool_target_size = self.cfg.importance_sampling_pool_size

        try:
            first_batch = next(inner_iterator)
        except StopIteration:
            return

        all_data = {key: [val] for key, val in first_batch.items() if val is not None}

        count_key = "pos" if "pos" in all_data else list(all_data.keys())[0]
        num_samples = len(all_data[count_key][0])

        pbar = tqdm(total=pool_target_size, desc="Building IS pool", unit="samples")
        pbar.update(num_samples)
        while num_samples < pool_target_size:
            try:
                batch = next(inner_iterator)
                for key, val in batch.items():
                    if val is not None:
                        all_data[key].append(val)
                added = len(batch[count_key])
                num_samples += added
                pbar.update(added)
            except StopIteration:
                hktex.debug("Inner dataset exhausted before pool size met.")
                break  # Handle non-infinite datasets
        pbar.close()

        # Concatenate all batches
        self.pool = {key: torch.cat(vals, dim=0) for key, vals in all_data.items()}

        # Trim pool
        self.pool_size = num_samples
        if self.pool_size > pool_target_size:
            for key in self.pool:
                self.pool[key] = self.pool[key][:pool_target_size]
            self.pool_size = pool_target_size

        self.errors = torch.ones(self.pool_size) * 1e5

    @torch.no_grad()
    def update_errors(self, indices: Tensor, per_point_loss: Tensor):
        """Updates the error buffer with an EMA."""
        current_errors = self.errors[indices.cpu()]
        new_errors = per_point_loss.cpu()

        beta = self.cfg.importance_sampling_ema_beta
        ema_errors = beta * current_errors + (1.0 - beta) * new_errors

        self.errors[indices] = ema_errors + 1e-6

        self.step += 1
        if self.step == self.cfg.importance_sampling_warmup_steps:
            if self.current_sampling_method == "uniform":
                self.current_sampling_method = "importance"
                hktex.debug("--- Switched to Importance Sampling ---")

    def __iter__(self):
        """The main iterator for Importance Sampling."""
        # Get batch_size from the inner dataset's config
        batch_size = self.inner_dataset.cfg.batch_size

        while True:
            if self.current_sampling_method == "uniform":
                indices = torch.randint(0, self.pool_size, (batch_size,))
            else:  # "importance"
                probs = self.errors / (self.errors.sum() + 1e-8)
                indices = torch.multinomial(probs, batch_size, replacement=True)

            batch = {key: self.pool[key][indices] for key in self.pool}
            batch["pool_indices"] = indices

            yield batch


class ImportanceSamplingMixIn:
    """
    A MixIn class that provides importance sampling train_dataloader
    and update_errors logic to a DataModule.

    It assumes 'self' will be an instance of MeshSamplerDataModule.
    """

    _is_wrapper_dataset: ImportanceSamplerDataset = None

    def train_dataloader(self: "MeshSamplerDataModule") -> DataLoader:
        if not hasattr(self, "train_dataset"):
            raise NotImplementedError(
                "Child datamodule did not create self.train_dataset in setup()"
            )

        if self.cfg.use_importance_sampling:
            if self._is_wrapper_dataset is None:
                hktex.debug("Wrapping dataset for Importance Sampling...")
                self._is_wrapper_dataset = ImportanceSamplerDataset(
                    inner_dataset=self.train_dataset,
                    cfg=self.cfg,
                )

            dataset_to_use = self._is_wrapper_dataset
        else:  # No importance sampling, use the original dataset!
            dataset_to_use = self.train_dataset

        return DataLoader(
            dataset_to_use,
            batch_size=None,
            num_workers=self.cfg.num_workers,
        )

    @torch.no_grad()
    def update_errors(self, indices: Tensor, per_point_loss: Tensor):
        if self._is_wrapper_dataset is not None:
            self._is_wrapper_dataset.update_errors(indices, per_point_loss)
