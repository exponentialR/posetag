"""Command-line entry point for the optional PoseTag workflow dashboard."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


INSTALL_HINT = 'pip install -e ".[gui]"'


class MissingGuiDependency(RuntimeError):
    """Raised when the optional PySide6 GUI dependency is unavailable."""


@dataclass(frozen=True)
class QtModules:
    """Container for lazily imported PySide6 modules."""

    QtCore: Any
    QtGui: Any
    QtWidgets: Any


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    """Build the ``posetag-gui`` argument parser without importing PySide6."""

    parser = argparse.ArgumentParser(
        prog="posetag-gui",
        formatter_class=_Formatter,
        description="Open the optional PoseTag workflow status dashboard.",
    )
    parser.add_argument(
        "--project_root",
        "--project-root",
        dest="project_root",
        default=".",
        help="PoseTag project root to inspect.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the optional PySide6 workflow dashboard."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_gui(args.project_root)
    except MissingGuiDependency as exc:
        print(str(exc), file=sys.stderr)
        print(f'Install the GUI extra with: {INSTALL_HINT}', file=sys.stderr)
        return 2


def run_gui(project_root: str | Path, qt_modules: QtModules | None = None) -> int:
    """Run the Qt application and return the Qt event-loop exit code."""

    qt = qt_modules or _load_qt_modules()
    argv = [sys.argv[0] if sys.argv else "posetag-gui"]
    app = qt.QtWidgets.QApplication.instance()
    if app is None:
        app = qt.QtWidgets.QApplication(argv)

    app.setApplicationName("PoseTag")
    app.setOrganizationName("PoseTag")

    from posetag.gui.main_window import build_main_window

    window = build_main_window(qt, Path(project_root).expanduser())
    window.resize(1180, 740)
    window.show()
    return int(app.exec())


def _load_qt_modules() -> QtModules:
    try:
        return QtModules(
            QtCore=importlib.import_module("PySide6.QtCore"),
            QtGui=importlib.import_module("PySide6.QtGui"),
            QtWidgets=importlib.import_module("PySide6.QtWidgets"),
        )
    except ImportError as exc:
        raise MissingGuiDependency(
            "PySide6 is required to launch the PoseTag GUI."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())

