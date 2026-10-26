"""Atomic four-file publication for controlled replay comparisons."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ReplayArtifactError(RuntimeError):
    """Base error for replay artifact validation and publication."""


class ReplayArtifactExistsError(ReplayArtifactError):
    """An output exists while overwrite is disabled."""


class ReplayArtifactPublicationError(ReplayArtifactError):
    """A complete replay pair could not be safely published."""


@dataclass(frozen=True, slots=True)
class ReplayArtifactPaths:
    """The four required, distinct outputs for one paired replay."""

    control_score: Path
    control_run: Path
    hierarchical_score: Path
    hierarchical_run: Path

    def in_publication_order(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.control_score,
            self.control_run,
            self.hierarchical_score,
            self.hierarchical_run,
        )


def preflight_replay_artifacts(
    paths: ReplayArtifactPaths,
    *,
    overwrite: bool,
) -> None:
    """Validate all destinations without creating directories or files."""

    if not isinstance(paths, ReplayArtifactPaths):
        raise ReplayArtifactPublicationError("replay artifact paths are invalid")
    destinations = paths.in_publication_order()
    if any(not isinstance(destination, Path) for destination in destinations):
        raise ReplayArtifactPublicationError("replay artifact paths are invalid")
    try:
        resolved = [destination.resolve() for destination in destinations]
    except OSError:
        raise ReplayArtifactPublicationError(
            "replay artifact paths are invalid"
        ) from None
    if len(set(resolved)) != len(resolved):
        raise ReplayArtifactPublicationError(
            "replay artifact paths must be distinct"
        )
    for destination in destinations:
        if destination.exists() and destination.is_dir():
            raise ReplayArtifactPublicationError(
                "replay artifact output path is a directory"
            )
        if destination.exists() and not overwrite:
            raise ReplayArtifactExistsError("replay artifact output already exists")


def publish_replay_artifacts(
    *,
    paths: ReplayArtifactPaths,
    control_run_json: str,
    control_scorecard_json: str,
    hierarchical_run_json: str,
    hierarchical_scorecard_json: str,
    overwrite: bool,
) -> None:
    """Publish one complete pair and roll back every handled partial write."""

    serialized_by_path = {
        paths.control_score: control_scorecard_json,
        paths.control_run: control_run_json,
        paths.hierarchical_score: hierarchical_scorecard_json,
        paths.hierarchical_run: hierarchical_run_json,
    }
    for serialized in serialized_by_path.values():
        _validate_serialized_json(serialized)
    preflight_replay_artifacts(paths, overwrite=overwrite)

    destinations = paths.in_publication_order()
    originals: dict[Path, bytes | None] = {}
    try:
        for destination in destinations:
            originals[destination] = (
                destination.read_bytes() if destination.exists() else None
            )
    except OSError:
        raise ReplayArtifactPublicationError(
            "replay artifact publication failed"
        ) from None

    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination in destinations:
            staged[destination] = _stage_text(
                destination,
                serialized_by_path[destination],
            )
        preflight_replay_artifacts(paths, overwrite=overwrite)
        for destination in destinations:
            os.replace(staged[destination], destination)
            published.append(destination)
    except Exception:
        rollback_failed = False
        for destination in reversed(published):
            try:
                original = originals[destination]
                if original is None:
                    destination.unlink(missing_ok=True)
                else:
                    _restore_bytes(destination, original)
            except Exception:
                rollback_failed = True
        if rollback_failed:
            raise ReplayArtifactPublicationError(
                "replay artifact publication and rollback failed"
            ) from None
        raise ReplayArtifactPublicationError(
            "replay artifact publication failed"
        ) from None
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _validate_serialized_json(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value.endswith("\n")
        or value.endswith("\n\n")
    ):
        raise ReplayArtifactPublicationError(
            "serialized replay artifact must end with exactly one newline"
        )
    try:
        json.loads(
            value,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ReplayArtifactPublicationError(
            "serialized replay artifact is invalid"
        ) from None


def _stage_text(destination: Path, value: str) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except OSError:
        raise ReplayArtifactPublicationError(
            "replay artifact staging failed"
        ) from None


def _restore_bytes(destination: Path, original: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.restore.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except Exception:
        raise ReplayArtifactPublicationError(
            "replay artifact restoration failed"
        ) from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
