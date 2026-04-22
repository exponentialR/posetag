from pathlib import Path
import os


def get_project_root(cli_value: Path | None) -> Path:
    """
    Decide the project root for data artefacts (boards/shots/objects/datasets).
    Priority: --project_root > $POSETAG_PROJECT > legacy $GTAT_PROJECT > CWD.
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    for env_name in ("POSETAG_PROJECT", "GTAT_PROJECT"):
        env = os.environ.get(env_name)
        if env:
            return Path(env).expanduser().resolve()
    return Path.cwd().resolve()

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
