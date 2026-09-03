from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_SPLITS = {"train", "val", "test"}


def ensure_split(split: str) -> str:
    if split not in VALID_SPLITS:
        raise ValueError(f"Split must be one of {sorted(VALID_SPLITS)}. Got: {split}")
    return split


def natural_step_number(path: Path) -> int:
    # Works for names like step12.frame_data.json and step12.camera.png.
    first_part = path.name.split(".", 1)[0]
    if not first_part.startswith("step"):
        return -1
    return int(first_part.replace("step", ""))


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            clear_directory_contents(child)
            child.rmdir()
