from pathlib import Path
import os

def get_project_root(cli_value: Path | None) -> Path:
    """
    Decide the project root for data artefacts (boards/shots/objects/datasets).
    Priority: --project_root > $GTAT_PROJECT > CWD.
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("GTAT_PROJECT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
