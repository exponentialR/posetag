"""PoseTag CLI wrapper for ChArUco camera calibration."""


def main():
    from utils.charuco_calibrate import main as _main

    return _main()
