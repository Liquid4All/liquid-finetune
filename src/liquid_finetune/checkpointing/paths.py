import os
import pathlib
import re
import shutil

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


_CLOUD_PREFIXES = ("s3://", "gs://", "az://", "abfs://", "abfss://", "hdfs://")


def is_local_path(path: str) -> bool:
    return not path.startswith(_CLOUD_PREFIXES)


def format_checkpoint_dir_name(
    run_name_template: str | None, epoch: float | None, step: int
) -> str:
    if not run_name_template:
        return f"{PREFIX_CHECKPOINT_DIR}-{step}"

    if "-" in run_name_template:
        base_part, time_part = run_name_template.rsplit("-", 1)
    else:
        base_part, time_part = run_name_template, ""

    epoch_num = int(epoch) if epoch else 0
    checkpoint_name = f"{base_part}-e{epoch_num}s{step}"
    if time_part:
        checkpoint_name += f"-{time_part}"
    return checkpoint_name


def resolve_checkpoint_output_dir(
    run_dir: str,
    run_name_template: str | None,
    epoch: float | None,
    step: int,
) -> str:
    return os.path.join(
        run_dir, format_checkpoint_dir_name(run_name_template, epoch, step)
    )


def current_checkpoint_output_dir(
    *,
    output_dir: str,
    run_name_template: str | None,
    epoch: float | None,
    step: int,
    manual_sharded: bool,
) -> str:
    """Return the checkpoint directory produced by the active save path."""
    if manual_sharded:
        return resolve_checkpoint_output_dir(
            run_dir=output_dir,
            run_name_template=run_name_template,
            epoch=epoch,
            step=step,
        )
    return os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def update_latest_pointer(run_dir: str, checkpoint_dir: str) -> None:
    latest_link = os.path.join(run_dir, "latest")
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
    if is_local_path(run_dir):
        try:
            if os.path.lexists(latest_link):
                os.unlink(latest_link)
            os.symlink(checkpoint_name, latest_link)
            return
        except OSError:
            pass

    with open(latest_link, "w") as handle:
        handle.write(f"{checkpoint_name}\n")


def rotate_named_checkpoints(output_dir: str | pathlib.Path, limit: int) -> None:
    checkpoints = []
    output_path = pathlib.Path(output_dir)
    for path in output_path.iterdir():
        if path.is_dir() and not path.name.startswith(".") and not path.is_symlink():
            match = re.search(r"-e(\d+)s(\d+)-", path.name)
            if match:
                checkpoints.append((int(match.group(2)), path))
                continue
            match = re.search(rf"{PREFIX_CHECKPOINT_DIR}-(\d+)", path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))

    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)


def rename_standard_checkpoint(
    *,
    output_dir: str,
    run_name_template: str | None,
    epoch: float | None,
    step: int,
    save_total_limit: int | None,
) -> pathlib.Path | None:
    """Rename a normal HF `checkpoint-N` directory to the Liquid run-name format."""
    if not run_name_template or not is_local_path(output_dir):
        return None

    source = pathlib.Path(output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{step}"
    if not source.exists():
        return None

    dest = pathlib.Path(
        resolve_checkpoint_output_dir(output_dir, run_name_template, epoch, step)
    )
    if dest.exists():
        return dest

    try:
        source.rename(dest)
    except OSError:
        return None

    update_latest_pointer(output_dir, str(dest))
    if save_total_limit is not None and save_total_limit > 0:
        rotate_named_checkpoints(output_dir, save_total_limit)
    return dest
