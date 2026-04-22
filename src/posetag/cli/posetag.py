#!/usr/bin/env python3
"""Primary PoseTag CLI for project management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from posetag.utils.project_config import (
    _norm,
    current_project_root,
    ensure_project_dirs,
    list_projects,
    projects_home_dir,
    register_project,
    set_current_project,
)


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def _cmd_project(args: argparse.Namespace) -> int:
    sub = args.project_cmd
    if sub == "ls":
        cur = current_project_root()
        projects = list_projects()
        if not projects:
            print("(no projects yet)")
            return 0
        for name, root in projects.items():
            mark = "*" if (cur and _norm(root) == _norm(cur)) else " "
            print(f"{mark} {name:20s} -> {root}")
        return 0

    if sub == "current":
        cur = current_project_root()
        if cur is None:
            print("(no current project)")
            return 1
        name = None
        for candidate, root in list_projects().items():
            if _norm(root) == _norm(cur):
                name = candidate
                break
        if name:
            print(f"{name} -> {cur}")
        else:
            print(str(cur))
        return 0

    if sub == "set":
        try:
            root = set_current_project(args.target)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"current -> {root}")
        return 0

    if sub == "new":
        if args.root:
            root = Path(args.root)
        else:
            home = projects_home_dir()
            if args.name:
                root = home / args.name
                if root.exists() and not any(root.iterdir()):
                    pass
            else:
                from datetime import datetime, timezone

                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
                root = home / f"project-{timestamp}"
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
    prog = Path(sys.argv[0]).name or "posetag"
    parser = argparse.ArgumentParser(
        prog,
        formatter_class=_Fmt,
        description="PoseTag project management commands.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    project = subparsers.add_parser("project", formatter_class=_Fmt, help="manage PoseTag projects")
    project_subparsers = project.add_subparsers(dest="project_cmd", required=True)

    project_ls = project_subparsers.add_parser("ls", help="list known projects")
    project_ls.set_defaults(func=_cmd_project)

    project_current = project_subparsers.add_parser("current", help="show current project")
    project_current.set_defaults(func=_cmd_project)

    project_set = project_subparsers.add_parser("set", help="set current project by NAME or PATH")
    project_set.add_argument("target", help="project NAME (registered) or PATH (existing dir)")
    project_set.set_defaults(func=_cmd_project)

    project_new = project_subparsers.add_parser("new", help="create and select a new project")
    project_new.add_argument("--name", help="human-readable name (also used as dirname when --root is omitted)")
    project_new.add_argument("--root", help="explicit project root (overrides --name location)")
    project_new.set_defaults(func=_cmd_project)

    project_home = project_subparsers.add_parser("home", help="print the PoseTag projects home directory")
    project_home.set_defaults(func=_cmd_project)

    args = parser.parse_args(argv)
    if args.cmd == "project":
        return _cmd_project(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
