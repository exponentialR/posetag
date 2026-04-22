import sys, runpy
from pathlib import Path


def run_legacy(script_relative_to_repo_root: str, argv=None):
    """
    Execute an existing top-level script by path, preserving CLI args.
    Retained only as a temporary compatibility helper during the PoseTag rename.
    """
    this = Path(__file__).resolve()
    # .../repo/src/gt6dof_atag/cli/_legacy.py -> repo root is parents[3]
    repo_root = this.parents[3]
    script_path = repo_root / script_relative_to_repo_root
    if argv is None:
        argv = sys.argv[1:]
    sys.argv = [str(script_path)] + list(argv)
    runpy.run_path(str(script_path), run_name="__main__")
