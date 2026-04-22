from __future__ import annotations
"""
Project resolution & registry for PoseTag.

Key behavior
------------
- Canonical projects home: ~/posetag.
- Legacy projects home: ~/gt-6dof (kept as a read fallback during migration).
- Priority when resolving a project root:
    1) CLI path (--project_root)
    2) $POSETAG_PROJECT (or legacy $GTAT_PROJECT)
    3) Config 'current' (name)
    4) Most recent existing project under the canonical/legacy projects homes
    5) Create a new timestamped project under ~/posetag
- Standard per-project layout: boards/, shots/, objects/, datasets/
- MRU tracking in ~/.config/posetag/config.yaml
- Legacy config in ~/.config/gt6dof_atag/config.yaml is read when the canonical
  config does not exist yet.
"""
import os, time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml

_APP_NAME = "posetag"
_LEGACY_APP_NAME = "gt6dof_atag"
_CFG_FILENAME = "config.yaml"
_PROJECTS_HOME_ENVS = ("POSETAG_PROJECTS_DIR", "GTAT_PROJECTS_DIR")
_ACTIVE_PROJECT_ENVS = ("POSETAG_PROJECT", "GTAT_PROJECT")
_DEFAULT_PROJECTS_HOME_NAME = "posetag"
_LEGACY_PROJECTS_HOME_NAME = "gt-6dof"
_STANDARD_SUBDIRS = ("boards", "shots", "objects", "datasets")


# --------- time & paths ---------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _config_dir() -> Path:
    """XDG on *nix/macOS; APPDATA on Windows; fallback to ~/.config/<app>."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / _APP_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / _APP_NAME


def config_path() -> Path:
    return _config_dir() / _CFG_FILENAME


def _legacy_config_path() -> Path:
    """Legacy config location retained as a read fallback during migration."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / _LEGACY_APP_NAME / _CFG_FILENAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / _LEGACY_APP_NAME / _CFG_FILENAME


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _norm(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()


# --------- projects home ---------
def _default_projects_homes() -> list[Path]:
    """Return canonical and legacy project homes in preferred discovery order."""
    canonical = _norm(Path.home() / _DEFAULT_PROJECTS_HOME_NAME)
    legacy = _norm(Path.home() / _LEGACY_PROJECTS_HOME_NAME)
    if canonical.exists():
        return [canonical] + ([legacy] if legacy != canonical else [])
    if legacy.exists():
        return [legacy, canonical]
    return [canonical]


def _env_projects_home() -> Optional[Path]:
    for env_name in _PROJECTS_HOME_ENVS:
        env_value = os.environ.get(env_name)
        if env_value:
            return _norm(env_value)
    return None


def projects_home_dir() -> Path:
    """
    Return the directory that houses all projects.
    Default: ~/posetag
    Override with $POSETAG_PROJECTS_DIR (or legacy $GTAT_PROJECTS_DIR).
    """
    home = _env_projects_home() or _default_projects_homes()[0]
    home.mkdir(parents=True, exist_ok=True)
    return home


def is_project_root(root: Path) -> bool:
    """
    A project root is recognized by the presence of at least one standard subdir,
    ideally all: boards/, shots/, objects/, datasets/.
    """
    root = _norm(root)
    found = [ (root / s).is_dir() for s in _STANDARD_SUBDIRS ]
    return any(found)


def ensure_project_dirs(root: Path) -> None:
    """Create the standard per-project layout under the given root."""
    root = _norm(root)
    for sub in _STANDARD_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


def _project_name_for(root: Path) -> str:
    """Default project name derived from the folder name."""
    return _norm(root).name


# --------- config I/O ---------
def load_config() -> Dict[str, Any]:
    primary = config_path()
    legacy = _legacy_config_path()
    path = primary if primary.exists() else legacy
    if not path.exists():
        return {"version": 1, "current": None, "projects": {}, "recent": []}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    cfg.setdefault("version", 1)
    cfg.setdefault("current", None)
    cfg.setdefault("projects", {})
    cfg.setdefault("recent", [])
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    _atomic_write(config_path(), yaml.safe_dump(cfg, sort_keys=True))


# --------- registry ---------
def register_project(root: Path, name: Optional[str] = None) -> str:
    """
    Register (or update) a project at 'root' with an optional human name.
    Ensures layout exists, updates MRU, returns the final project name.
    """
    root = _norm(root)
    ensure_project_dirs(root)

    cfg = load_config()
    name = name or _project_name_for(root)

    # ensure unique name if colliding with a different root
    base = name
    i = 2
    while name in cfg["projects"] and _norm(cfg["projects"][name]["root"]) != root:
        name = f"{base}-{i}"
        i += 1

    prev = cfg["projects"].get(name, {})
    cfg["projects"][name] = {
        "root": str(root),
        "created": prev.get("created", _now_iso()),
        "updated": _now_iso(),
    }
    cfg["current"] = name
    recent = [n for n in cfg.get("recent", []) if n != name]
    cfg["recent"] = [name] + recent[:9]
    save_config(cfg)
    return name


def set_current_project(name_or_path: str | Path) -> Path:
    """
    Set the current project either by name or by existing path.
    Returns the normalized project root.
    """
    p = Path(str(name_or_path))
    if p.exists():
        # Treat as a path
        name = register_project(_norm(p))
        # Reload to reflect register_project changes
        cfg2 = load_config()
        return _norm(Path(cfg2["projects"][name]["root"]))

    # Treat as a name
    cfg = load_config()
    name = str(name_or_path)
    if name not in cfg["projects"]:
        raise ValueError(f"Unknown project name: {name}")
    cfg["current"] = name
    cfg["projects"][name]["updated"] = _now_iso()
    cfg["recent"] = [name] + [n for n in cfg.get("recent", []) if n != name][:9]
    save_config(cfg)
    return _norm(Path(cfg["projects"][name]["root"]))


def list_projects() -> Dict[str, Path]:
    """Return {name: root_path} for all known projects."""
    cfg = load_config()
    return {name: _norm(Path(info["root"])) for name, info in cfg["projects"].items()}


def current_project_root() -> Optional[Path]:
    """Return the root of the current project (if any)."""
    cfg = load_config()
    cur = cfg.get("current")
    if not cur:
        return None
    info = cfg["projects"].get(cur)
    if not info:
        return None
    return _norm(Path(info["root"]))


# --------- discovery under project homes ---------
def _iter_home_projects(home: Path) -> List[Tuple[Path, float]]:
    """
    Scan the projects home for candidate project roots, returning [(path, mtime)].
    Recognizes subdirectories that look like a project (see is_project_root).
    """
    home = _norm(home)
    if not home.exists():
        return []
    out: List[Tuple[Path, float]] = []
    for child in home.iterdir():
        if child.is_dir() and is_project_root(child):
            try:
                mtime = child.stat().st_mtime
            except Exception:
                mtime = 0.0
            out.append((_norm(child), mtime))
    # newest (largest mtime) first
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _new_default_project_root(home: Path) -> Path:
    """
    Create a new timestamped project directory under projects home.
    Example: ~/posetag/project-2025-11-08T12-34-56Z
    """
    ts = _now_iso().replace(":", "-")
    candidate = home / f"project-{ts}"
    candidate.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs(candidate)
    return _norm(candidate)


# --------- resolution ---------
def resolve_project_root(cli: Optional[Path | str] = None) -> Path:
    """
    Resolve the active project root with the following priority:
      1) CLI (--project_root) path if provided -> register & return
      2) $POSETAG_PROJECT (or legacy $GTAT_PROJECT) -> register & return
      3) config.current (name) if set -> ensure layout & return
      4) most recent project under the canonical/legacy projects homes -> register & return
      5) create a new timestamped project under projects_home_dir() -> register & return
    """
    # 1) CLI
    if cli:
        root = _norm(cli)
        register_project(root)
        return root

    # 2) ENV path
    for env_name in _ACTIVE_PROJECT_ENVS:
        env = os.environ.get(env_name)
        if env:
            root = _norm(env)
            register_project(root)
            return root

    # 3) Config 'current'
    cur = current_project_root()
    if cur:
        ensure_project_dirs(cur)
        return cur

    # 4) Most recent under canonical/legacy project homes
    homes = [projects_home_dir()]
    for candidate in _default_projects_homes():
        if candidate not in homes:
            homes.append(candidate)
    for home in homes:
        discovered = _iter_home_projects(home)
        if discovered:
            root = discovered[0][0]
            register_project(root)  # update MRU/current
            return root

    # 5) Create a new project under the preferred projects home
    root = _new_default_project_root(projects_home_dir())
    register_project(root)
    return root
