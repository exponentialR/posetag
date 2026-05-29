"""PoseTag CLI wrapper for ChArUco camera calibration."""


def main(argv=None):
    from utils.charuco_calibrate import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
