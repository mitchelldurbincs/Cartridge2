"""Immutable checkpoint publication for the standalone learner."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import metrics as prom_metrics
from .checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from .storage.publisher import CheckpointRef

if TYPE_CHECKING:
    from .trainer import AlphaZeroLearner

logger = logging.getLogger(__name__)


def handle_checkpoint(trainer: "AlphaZeroLearner", step: int, global_step: int) -> None:
    """Publish the immutable checkpoint due at this step."""
    if trainer.config.defer_run_commit:
        # The parent orchestrator must receive one final direct child of the
        # inherited RunHead checkpoint. Intermediate staged descendants would
        # be invisible orphans and make that final transition invalid.
        return

    checkpoint: CheckpointRef | None = None
    if step % trainer.config.checkpoint_interval == 0:
        started = time.time()
        checkpoint = save_checkpoint(trainer, global_step)
        prom_metrics.record_checkpoint(time.time() - started)
        trainer.stats.last_checkpoint = checkpoint.checkpoint_id
        logger.info("Staged checkpoint: %s", checkpoint.checkpoint_id)

    if checkpoint is not None:
        trainer._publish_run_state(checkpoint)


def save_checkpoint(trainer: "AlphaZeroLearner", step: int) -> CheckpointRef:
    """Materialize validated blobs and manifest without advancing RunHeadV2."""
    if (
        trainer.last_checkpoint_ref is not None
        and trainer.last_checkpoint_ref.manifest.step == step
    ):
        return trainer.last_checkpoint_ref
    model_root = Path(trainer.config.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=model_root) as staging:
        staging_dir = Path(staging)
        learner_state_contract = LearnerStateContract(
            trainer.network,
            trainer.optimizer,
            trainer.lr_scheduler,
        )
        onnx_path = export_onnx_artifact(
            network=trainer.network,
            output_path=staging_dir / "model.onnx",
            device=trainer.device,
            artifact_contract=trainer.artifact_contract,
        )
        learner_path = write_learner_state_artifact(
            network=trainer.network,
            optimizer=trainer.optimizer,
            step=step,
            output_path=staging_dir / "learner.pt",
            artifact_contract=trainer.artifact_contract,
            config_sha256=trainer.config_sha256,
            scheduler=trainer.lr_scheduler,
        )
        checkpoint = trainer.checkpoint_publisher.stage_checkpoint(
            onnx_path,
            learner_path,
            step=step,
            parent_checkpoint_id=trainer.parent_checkpoint_id,
            config_sha256=trainer.config_sha256,
            learner_state_contract=learner_state_contract,
        )
    # Lineage advances only after RunHead selects this checkpoint. Deferred
    # loops leave that transition to their parent orchestrator.
    trainer.last_checkpoint_ref = checkpoint
    return checkpoint


__all__ = ["handle_checkpoint", "save_checkpoint"]
