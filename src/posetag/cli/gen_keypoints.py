"""PoseTag CLI wrapper for annotation-ready mesh keypoint generation."""


def main(argv=None):
    from gen_keypoints import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
