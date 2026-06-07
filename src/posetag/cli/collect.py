"""PoseTag CLI wrapper for dataset collection."""


def main(argv=None):
    from collect_gt_dataset import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
