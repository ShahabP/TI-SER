"""Command-line entry point (``ti-ser`` / ``python -m ti_ser``)."""
from typing import Optional, Sequence

from .runner import parse_args, run_experiments


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_experiments(args)


if __name__ == "__main__":
    main()
