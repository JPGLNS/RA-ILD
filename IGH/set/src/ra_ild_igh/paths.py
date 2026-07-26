#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository and project-path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


class RepositoryRootError(RuntimeError):
    """Raised when the RA-ILD repository root cannot be located."""


def _looks_like_root(path: Path) -> bool:
    if (path / ".git").exists():
        return True
    return (
        (path / ".gitignore").is_file()
        and (path / "IGH").is_dir()
        and (path / "IGH").exists()
    )


def find_repository_root(start: Optional[PathLike] = None) -> Path:
    """Find the repository root by walking upward from ``start`` or this file."""
    starts = []
    if start is not None:
        candidate = Path(start).expanduser()
        starts.append(candidate.parent if candidate.is_file() else candidate)
    starts.extend([Path.cwd(), Path(__file__).resolve().parent])

    inspected = []
    for initial in starts:
        initial = initial.resolve()
        for candidate in (initial, *initial.parents):
            inspected.append(str(candidate))
            if _looks_like_root(candidate):
                return candidate

    raise RepositoryRootError(
        "Could not locate the RA-ILD repository root. Inspected:\n  - "
        + "\n  - ".join(dict.fromkeys(inspected))
    )


def resolve_project_path(
    value: PathLike,
    repository_root: PathLike,
    *,
    must_exist: bool = False,
    expect: Optional[str] = None,
) -> Path:
    """Resolve a repository-relative or absolute configured path."""
    root = Path(repository_root).expanduser().resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Configured path does not exist: {path}")
    if expect == "file" and path.exists() and not path.is_file():
        raise IsADirectoryError(f"Expected a file: {path}")
    if expect == "dir" and path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")
    if expect not in {None, "file", "dir"}:
        raise ValueError("expect must be None, 'file', or 'dir'")
    return path
