"""PoseTag CLI wrapper for face-shot capture."""


def main(argv=None):
    from capture_face import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
