"""Legacy compatibility wrapper for AprilTag sheet generation."""

from posetag.pipelines.generate_tags import main


if __name__ == "__main__":
    raise SystemExit(main())
