#!/usr/bin/env python3
"""
gtat – tiny CLI for GT-6DoF-ATag project management.

Usage:
  gtat project ls
  gtat project current
  gtat project set <name-or-path>
  gtat project new [--name NAME] [--root PATH]
  gtat project home
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from gt6dof_atag.utils.project_config import (
    list_projects,
    current_project_root,
    set_current_project,
    register_project,
    projects_home_dir,
    ensure_project_dirs,
    _norm,  # optional, but handy for consistent printing
)

class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def _cmd_project(args: argparse.Namespace) -> int:
    sub = args.project_cmd
    if sub == "ls":
        cur = current_project_root()
        projects = list_projects()
        if not projects:
            print("(no projects yet)"); return 0
        for name, root in projects.items():
            mark = "*" if (cur and _norm(root) == _norm(cur)) else " "
            print(f"{mark} {name:20s} -> {root}")
        return 0

    if sub == "current":
        cur = current_project_root()
        if cur is None:
            print("(no current project)"); return 1
        # find name if registered
        name = None
        for n, r in list_projects().items():
            if _norm(r) == _norm(cur):
                name = n; break
        if name:
            print(f"{name} -> {cur}")
        else:
            print(str(cur))
        return 0

    if sub == "set":
        target = args.target
        try:
            root = set_current_project(target)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"current -> {root}")
        return 0

    if sub == "new":
        # Decide root
        if args.root:
            root = Path(args.root)
        else:
            home = projects_home_dir()
            if args.name:
                root = home / args.name
                # avoid accidental overwrite
                if root.exists() and not any(root.iterdir()):
                    pass  # empty dir is fine
            else:
                # let project_config create a timestamped folder if needed; but here we use NAME or ROOT path
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
                root = home / f"project-{ts}"
        ensure_project_dirs(root)
        name = register_project(root, name=args.name)
        print(f"created: {name} -> {root}")
        return 0

    if sub == "home":
        print(projects_home_dir())
        return 0

    print("unknown subcommand", file=sys.stderr)
    return 2

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("gtat", formatter_class=_Fmt)
    sp = p.add_subparsers(dest="cmd", required=True)

    pj = sp.add_parser("project", formatter_class=_Fmt, help="manage projects")
    pj_sp = pj.add_subparsers(dest="project_cmd", required=True)

    pj_ls = pj_sp.add_parser("ls", help="list known projects")
    pj_ls.set_defaults(func=_cmd_project)

    pj_cur = pj_sp.add_parser("current", help="show current project")
    pj_cur.set_defaults(func=_cmd_project)

    pj_set = pj_sp.add_parser("set", help="set current project by NAME or PATH")
    pj_set.add_argument("target", help="project NAME (registered) or PATH (existing dir)")
    pj_set.set_defaults(func=_cmd_project)

    pj_new = pj_sp.add_parser("new", help="create and select a new project")
    pj_new.add_argument("--name", help="human-readable name (also used as dirname when --root not given)")
    pj_new.add_argument("--root", help="explicit project root (overrides --name location)")
    pj_new.set_defaults(func=_cmd_project)

    pj_home = pj_sp.add_parser("home", help="print the projects home directory")
    pj_home.set_defaults(func=_cmd_project)

    args = p.parse_args(argv)
    if args.cmd == "project":
        return _cmd_project(args)
    p.print_help(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
