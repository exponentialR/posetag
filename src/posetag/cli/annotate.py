"""PoseTag CLI wrapper for per-face annotation."""


def main(argv=None):
    from annotate_shots import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
