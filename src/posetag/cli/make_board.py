"""PoseTag CLI wrapper for AprilTag board creation."""


def main(argv=None):
    from make_board import main as _main

    return _main(argv)
