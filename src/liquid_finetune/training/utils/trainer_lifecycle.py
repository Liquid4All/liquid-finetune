import logging

logger = logging.getLogger(__name__)

_DISTRIBUTED_CLEANUP_KEYWORDS = (
    "cuda error",
    "ecc error",
    "nccl",
    "collective",
)


def trainer_reached_planned_end(trainer) -> bool:
    """Return true when Trainer state has reached configured steps or epochs."""
    state = getattr(trainer, "state", None)
    args = getattr(trainer, "args", None)
    if state is None or args is None:
        return False

    max_steps = int(getattr(args, "max_steps", -1) or -1)
    global_step = int(getattr(state, "global_step", 0) or 0)
    if max_steps > 0:
        return global_step >= max_steps

    epoch = getattr(state, "epoch", None)
    num_train_epochs = getattr(args, "num_train_epochs", None)
    if epoch is None or num_train_epochs is None:
        return False
    return float(epoch) >= float(num_train_epochs)


def run_training_safely(trainer, **kwargs):
    """Run trainer.train() and suppress only post-target distributed teardown errors."""
    try:
        trainer.train(**kwargs)
        logger.info("Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e).lower()
        is_cleanup_error = trainer_reached_planned_end(trainer) and any(
            kw in error_msg for kw in _DISTRIBUTED_CLEANUP_KEYWORDS
        )
        if is_cleanup_error:
            logger.warning(
                "Training completed but hit distributed communication error "
                "during cleanup: %s",
                e,
            )
        else:
            raise
