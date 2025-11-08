from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

_APP_NAME = "gt6dof_atag"
_CFG_FILENAME = "config.yaml"

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _config_dir() -> Path:
    # XDG on Linux/macOS; APPDATA on Windows; fallback to ~/.config
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / _APP_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / _APP_NAME

def config_path() -> Path:
    return _config_dir() / _CFG_FILENAME

def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)

def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {"version": 1, "current": None, "projects": {}, "recent": []}
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    cfg.setdefault("version", 1)
    cfg.setdefault("current", None)
    cfg.setdefault("projects", {})
    cfg.setdefault("recent", [])
    return cfg

def save_config(cfg: Dict[str, Any]) -> None:
    _atomic_write(config_path(), yaml.safe_dump(cfg, sort_keys=True))

def _norm(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()

def ensure_project_dirs(root: Path) -> None:
    # Standard layout under a project root
    for sub in ("boards", "shots", "objects", "datasets"):
        (root / sub).mkdir(parents=True, exist_ok=True)

def _project_name_for(root: Path) -> str:
    # default name from folder
    return root.name

def register_project(root: Path, name: Optional[str] = None) -> str:
    root = _norm(root)
    ensure_project_dirs(root)
    cfg = load_config()

    name = name or _project_name_for(root)
    # ensure unique name
    base = name
    i = 2
    while name in cfg["projects"] and _norm(cfg["projects"][name]["root"]) != root:
        name = f"{base}-{i}"
        i += 1

    cfg["projects"][name] = {
        "root": str(root),
        "created": cfg["projects"].get(name, {}).get("created", _now_iso()),
        "updated": _now_iso(),
    }
    # set current + recent MRU
    cfg["current"] = name
    recent = [n for n in cfg.get("recent", []) if n != name]
    cfg["recent"] = [name] + recent[:9]
    save_config(cfg)
    return name

def set_current_project(name_or_path: str | Path) -> Path:
    cfg = load_config()
    # path given?
    p = Path(str(name_or_path))
    if Path(name_or_path).exists():
        name = register_project(_norm(p))
        return _norm(Path(cfg["projects"][name]["root"]))
    # name given
    name = str(name_or_path)
    if name not in cfg["projects"]:
        raise ValueError(f"Unknown project name: {name}")
    cfg["current"] = name
    cfg["projects"][name]["updated"] = _now_iso()
    cfg["recent"] = [name] + [n for n in cfg.get("recent", []) if n != name][:9]
    save_config(cfg)
    return _norm(Path(cfg["projects"][name]["root"]))

def list_projects() -> Dict[str, Path]:
    cfg = load_config()
    return {name: _norm(Path(info["root"])) for name, info in cfg["projects"].items()}

def current_project_root() -> Optional[Path]:
    cfg = load_config()
    cur = cfg.get("current")
    if not cur:
        return None
    info = cfg["projects"].get(cur)
    if not info:
        return None
    return _norm(Path(info["root"]))

def resolve_project_root(cli: Optional[Path | str] = None) -> Path:
    """
    Priority:
      1) --project_root CLI (path) if provided
      2) $GTAT_PROJECT (path)
      3) config.current (name) if set
      4) default = CWD / 'gtat_project'
    Registers and makes it current; ensures standard subdirs exist.
    """
    # 1) CLI
    if cli:
        root = _norm(cli)
        register_project(root)
        return root
    # 2) ENV
    env = os.environ.get("GTAT_PROJECT")
    if env:
        root = _norm(env)
        register_project(root)
        return root
    # 3) Config current
    cur = current_project_root()
    if cur:
        ensure_project_dirs(cur)
        return cur
    # 4) Default
    default = _norm(Path.cwd() / "gtat_project")
    register_project(default)
    return default
