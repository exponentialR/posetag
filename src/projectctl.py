import argparse
from pathlib import Path
from utils.project_config import (
    load_config, save_config, list_projects, register_project,
    set_current_project, current_project_root, resolve_project_root
)

def main():
    ap = argparse.ArgumentParser("Project control")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("current", help="Show current project")
    sub.add_parser("list", help="List known projects")
    p_use = sub.add_parser("use", help="Switch current project")
    p_use.add_argument("name_or_path")

    p_add = sub.add_parser("add", help="Register a project")
    p_add.add_argument("path", type=Path)
    p_add.add_argument("--name")

    p_rm = sub.add_parser("remove", help="Forget a project by name")
    p_rm.add_argument("name")

    args = ap.parse_args()

    if args.cmd == "current":
        root = current_project_root()
        print(root if root else "(none)")
    elif args.cmd == "list":
        for name, root in list_projects().items():
            cur = "*" if current_project_root() == root else " "
            print(f"{cur} {name:20s}  {root}")
    elif args.cmd == "use":
        root = set_current_project(args.name_or_path)
        print(root)
    elif args.cmd == "add":
        name = register_project(args.path, args.name)
        print(f"added: {name}")
    elif args.cmd == "remove":
        cfg = load_config()
        if args.name in cfg["projects"]:
            del cfg["projects"][args.name]
            if cfg.get("current") == args.name:
                cfg["current"] = None
            cfg["recent"] = [n for n in cfg.get("recent", []) if n != args.name]
            save_config(cfg)
            print("removed")
        else:
            print("no such project")

if __name__ == "__main__":
    main()
