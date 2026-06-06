"""Workflow helpers for launching PoseTag face annotation from the GUI."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

from posetag.workflows.status import WorkflowStatus, inspect_project

ANNOTATION_PROCESS_NOT_STARTED = "not_started"
ANNOTATION_PROCESS_RUNNING = "running"
ANNOTATION_PROCESS_FINISHED = "finished"
ANNOTATION_PROCESS_FAILED = "failed"

ANNOTATION_MODE_BROWSE = "browse"
ANNOTATION_MODE_BATCH = "batch"
ANNOTATION_MODE_DRY_RUN = "dry-run"


@dataclass(frozen=True)
class AnnotationOutputStatus:
    manifest_path: Path
    face_manifest_path: Path
    expected_yaml_paths: tuple[Path, ...]
    existing_yaml_paths: tuple[Path, ...]
    missing_yaml_paths: tuple[Path, ...]
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def expected_count(self) -> int:
        return len(self.expected_yaml_paths)

    @property
    def existing_count(self) -> int:
        return len(self.existing_yaml_paths)

    @property
    def missing_count(self) -> int:
        return len(self.missing_yaml_paths)


@dataclass(frozen=True)
class AnnotationReadiness:
    ready: bool
    command_preview: str
    dry_run_command_preview: str
    batch_command_preview: str
    output_status: AnnotationOutputStatus
    checked_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotationLaunch:
    program: str
    arguments: tuple[str, ...]
    display_command: str
    project_root: Path
    expected_face_manifest: Path
    mode: str


@dataclass(frozen=True)
class AnnotationProcessState:
    state: str
    label: str
    message: str
    running: bool = False
    success: bool = False
    expected_face_manifest: Path | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class RemovedAnnotationOutput:
    yaml_path: Path
    removed: bool
    face_manifest_path: Path


def inspect_annotation_readiness(
    project_root: Union[Path, str],
) -> AnnotationReadiness:
    """Return launch readiness and expected annotation outputs."""

    root = Path(project_root).expanduser()
    manifest_path = root / "shots" / "manifest.csv"
    face_manifest_path = root / "faces" / "face_manifest.csv"
    stages = tuple(inspect_project(root))
    stage6 = next((stage for stage in stages if stage.stage_id == 6), None)
    stage7 = next((stage for stage in stages if stage.stage_id == 7), None)

    stage7_checked_paths = tuple(
        Path(path) for path in (stage7.checked_paths if stage7 is not None else ())
    )
    expected = tuple(
        path
        for path in stage7_checked_paths
        if path.name.endswith("_T_board_object.yaml")
    )
    existing = tuple(path for path in expected if path.exists())
    missing = tuple(path for path in expected if not path.exists())
    checked_paths = tuple(
        dict.fromkeys(
            (
                manifest_path,
                face_manifest_path,
                *stage7_checked_paths,
            )
        )
    )
    warnings = list(stage7.warnings if stage7 is not None else ())
    if stage7 is not None:
        warnings.extend(stage7.errors)
    errors: list[str] = []

    if stage6 is not None and stage6.status != WorkflowStatus.COMPLETE:
        errors.append("Complete Stage 6 mesh keypoints before annotation.")
    if stage7 is None:
        errors.append("Stage 7 status could not be inspected.")
    elif stage7.status == WorkflowStatus.NOT_APPLICABLE:
        errors.append(stage7.message)
    if not manifest_path.exists():
        errors.append(f"shots/manifest.csv not found: {manifest_path}")

    output_status = AnnotationOutputStatus(
        manifest_path=manifest_path,
        face_manifest_path=face_manifest_path,
        expected_yaml_paths=expected,
        existing_yaml_paths=existing,
        missing_yaml_paths=missing,
        checked_paths=checked_paths,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )

    return AnnotationReadiness(
        ready=not errors,
        command_preview=annotation_command_preview(root, ANNOTATION_MODE_BROWSE),
        dry_run_command_preview=annotation_command_preview(root, ANNOTATION_MODE_DRY_RUN),
        batch_command_preview=annotation_command_preview(root, ANNOTATION_MODE_BATCH),
        output_status=output_status,
        checked_paths=checked_paths,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def build_annotation_launch(
    project_root: Union[Path, str],
    *,
    mode: str = ANNOTATION_MODE_BROWSE,
    pts_type: str = "quad",
    check_tag_scale: bool = False,
    auto_correct_scale: bool = False,
    force: bool = False,
    scale_tol: float | None = None,
) -> AnnotationLaunch:
    """Build an editable-install-safe annotation launch command."""

    normalized_mode = _normalize_annotation_mode(mode)
    root = Path(project_root).expanduser()
    args = ["-m", "posetag.cli.annotate"]
    if normalized_mode == ANNOTATION_MODE_BROWSE:
        args.append("--browse")
    elif normalized_mode == ANNOTATION_MODE_BATCH:
        args.extend(["--batch", "latest"])
    elif normalized_mode == ANNOTATION_MODE_DRY_RUN:
        args.extend(["--batch", "latest", "--dry-run"])
    args.extend(_annotation_option_args(
        pts_type=pts_type,
        check_tag_scale=check_tag_scale,
        auto_correct_scale=auto_correct_scale,
        force=force,
        scale_tol=scale_tol,
    ))

    return AnnotationLaunch(
        program=sys.executable,
        arguments=tuple(args),
        display_command=annotation_command_preview(
            root,
            normalized_mode,
            pts_type=pts_type,
            check_tag_scale=check_tag_scale,
            auto_correct_scale=auto_correct_scale,
            force=force,
            scale_tol=scale_tol,
        ),
        project_root=root,
        expected_face_manifest=root / "faces" / "face_manifest.csv",
        mode=normalized_mode,
    )


def annotation_command_preview(
    project_root: Union[Path, str],
    mode: str = ANNOTATION_MODE_BROWSE,
    *,
    pts_type: str = "quad",
    check_tag_scale: bool = False,
    auto_correct_scale: bool = False,
    force: bool = False,
    scale_tol: float | None = None,
) -> str:
    """Return the user-facing command equivalent for an annotation launch."""

    normalized_mode = _normalize_annotation_mode(mode)
    root = str(Path(project_root).expanduser())
    parts: list[str] = ["env", f"POSETAG_PROJECT={root}", "posetag-annotate"]
    if normalized_mode == ANNOTATION_MODE_BROWSE:
        parts.append("--browse")
    elif normalized_mode == ANNOTATION_MODE_BATCH:
        parts.extend(["--batch", "latest"])
    elif normalized_mode == ANNOTATION_MODE_DRY_RUN:
        parts.extend(["--batch", "latest", "--dry-run"])
    parts.extend(_annotation_option_args(
        pts_type=pts_type,
        check_tag_scale=check_tag_scale,
        auto_correct_scale=auto_correct_scale,
        force=force,
        scale_tol=scale_tol,
    ))
    return shlex.join(parts)


def annotation_process_not_started(
    expected_face_manifest: Union[Path, str, None] = None,
) -> AnnotationProcessState:
    path = Path(expected_face_manifest).expanduser() if expected_face_manifest else None
    return AnnotationProcessState(
        state=ANNOTATION_PROCESS_NOT_STARTED,
        label="not started",
        message=(
            "Open the annotator when Stage 6 keypoints and captured face shots "
            "are ready."
        ),
        expected_face_manifest=path,
    )


def annotation_process_running(
    expected_face_manifest: Union[Path, str, None] = None,
    *,
    mode: str = ANNOTATION_MODE_BROWSE,
) -> AnnotationProcessState:
    path = Path(expected_face_manifest).expanduser() if expected_face_manifest else None
    normalized_mode = _normalize_annotation_mode(mode)
    action = (
        "Use the OpenCV browser to choose a face and press ENTER to annotate."
        if normalized_mode == ANNOTATION_MODE_BROWSE
        else "Use the OpenCV annotation windows to click and accept each face."
    )
    if normalized_mode == ANNOTATION_MODE_DRY_RUN:
        action = "Dry-run is validating annotation inputs without opening the click UI."
    return AnnotationProcessState(
        state=ANNOTATION_PROCESS_RUNNING,
        label="running",
        message=action,
        running=True,
        expected_face_manifest=path,
    )


def annotation_process_failed(
    message: str,
    *,
    expected_face_manifest: Union[Path, str, None] = None,
    exit_code: int | None = None,
) -> AnnotationProcessState:
    path = Path(expected_face_manifest).expanduser() if expected_face_manifest else None
    return AnnotationProcessState(
        state=ANNOTATION_PROCESS_FAILED,
        label="failed",
        message=message,
        expected_face_manifest=path,
        exit_code=exit_code,
    )


def summarize_annotation_process_result(
    *,
    exit_code: int,
    crashed: bool,
    expected_face_manifest: Union[Path, str, None] = None,
) -> AnnotationProcessState:
    path = Path(expected_face_manifest).expanduser() if expected_face_manifest else None
    if crashed:
        return annotation_process_failed(
            "Annotation process crashed before completing.",
            expected_face_manifest=path,
            exit_code=exit_code,
        )
    if exit_code != 0:
        return annotation_process_failed(
            "Annotation process exited with an error. Check the process log.",
            expected_face_manifest=path,
            exit_code=exit_code,
        )
    return AnnotationProcessState(
        state=ANNOTATION_PROCESS_FINISHED,
        label="finished",
        message=(
            "Annotation process exited. Stage 7 status was refreshed from "
            "the saved annotation YAMLs and face manifest."
        ),
        success=True,
        expected_face_manifest=path,
        exit_code=exit_code,
    )


def _normalize_annotation_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("_", "-")
    aliases = {
        "browse": ANNOTATION_MODE_BROWSE,
        "browser": ANNOTATION_MODE_BROWSE,
        "batch": ANNOTATION_MODE_BATCH,
        "latest": ANNOTATION_MODE_BATCH,
        "dry-run": ANNOTATION_MODE_DRY_RUN,
        "dryrun": ANNOTATION_MODE_DRY_RUN,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(
            (ANNOTATION_MODE_BROWSE, ANNOTATION_MODE_BATCH, ANNOTATION_MODE_DRY_RUN)
        )
        raise ValueError(f"Unsupported annotation launch mode {mode!r}. Use {supported}.") from exc


def _annotation_option_args(
    *,
    pts_type: str = "quad",
    check_tag_scale: bool = False,
    auto_correct_scale: bool = False,
    force: bool = False,
    scale_tol: float | None = None,
) -> list[str]:
    args: list[str] = []
    normalized_pts_type = str(pts_type or "quad").strip().lower()
    if normalized_pts_type not in {"quad", "any"}:
        raise ValueError("Unsupported annotation corner mode. Use 'quad' or 'any'.")
    if normalized_pts_type != "quad":
        args.extend(["--pts-type", normalized_pts_type])
    if check_tag_scale:
        args.append("--check-tag-scale")
    if auto_correct_scale:
        args.append("--auto-correct-scale")
    if force:
        args.append("--force")
    if scale_tol is not None:
        args.extend(["--scale-tol", f"{float(scale_tol):g}"])
    return args


def remove_annotation_output(
    project_root: Union[Path, str],
    yaml_path: Union[Path, str],
) -> RemovedAnnotationOutput:
    """Remove one Stage 7 board-to-object transform and refresh face manifest."""

    root = Path(project_root).expanduser().resolve()
    faces_root = (root / "faces").resolve()
    target = _resolve_annotation_output_path(root, faces_root, yaml_path)
    if faces_root != target and faces_root not in target.parents:
        raise ValueError("Annotation output must be inside the project faces directory.")
    if not target.name.endswith("_T_board_object.yaml"):
        raise ValueError("Annotation output must be a *_T_board_object.yaml file.")

    removed = False
    if target.exists():
        target.unlink()
        removed = True

    try:
        import annotate_shots

        annotate_shots._sync_face_manifest(root)
    except Exception as exc:  # pragma: no cover - exercised via GUI failure path.
        raise RuntimeError(f"Removed transform but could not refresh face manifest: {exc}") from exc

    return RemovedAnnotationOutput(
        yaml_path=target,
        removed=removed,
        face_manifest_path=root / "faces" / "face_manifest.csv",
    )


def _resolve_annotation_output_path(
    project_root: Path,
    faces_root: Path,
    yaml_path: Union[Path, str],
) -> Path:
    raw = Path(yaml_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates: list[Path] = [
        (Path.cwd() / raw).resolve(),
        (project_root / raw).resolve(),
    ]
    parts = raw.parts
    if parts and parts[0] == project_root.name:
        candidates.append((project_root.parent / raw).resolve())
        if len(parts) > 1:
            candidates.append((project_root / Path(*parts[1:])).resolve())

    for candidate in candidates:
        if faces_root == candidate or faces_root in candidate.parents:
            return candidate
    return candidates[0]


def face_labels_from_annotation_paths(
    project_root: Union[Path, str],
    paths: Iterable[Path],
) -> tuple[str, ...]:
    """Return compact labels such as ``column_white / front`` for YAML paths."""

    root = Path(project_root).expanduser()
    labels: list[str] = []
    for path in paths:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        parts = rel.parts
        if len(parts) >= 4 and parts[0] == "faces":
            labels.append(f"{parts[1]} / {parts[2]}")
        else:
            labels.append(path.stem.replace("_T_board_object", ""))
    return tuple(labels)
