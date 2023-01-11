"""A cli entry point."""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mfpbench
import mfpbench.download
import mfpbench.priors


@dataclass
class CommandHandler(ABC):
    """A handler for a command."""

    @abstractmethod
    def __call__(self, args: argparse.Namespace) -> None:
        """Handle the command."""
        ...

    @property
    @abstractmethod
    def parser(self) -> argparse.ArgumentParser:
        """Parser for the command."""
        ...


class DownloadHandler(CommandHandler):
    def __call__(self, args: argparse.Namespace) -> None:
        """Download the data."""
        mfpbench.download.download(datadir=args.data_dir, force=args.force)

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force download and remove existing data",
        )
        parser.add_argument("--data-dir", type=str, help="Where to save the data")
        return parser


class GeneratePriorsHandler(CommandHandler):
    def __call__(self, args: argparse.Namespace) -> None:
        mfpbench.priors.generate_priors(
            seed=args.seed,
            nsamples=args.nsamples,
            prefix=args.prefix,
            to=args.to,
            fidelity=args.fidelity,
            only=args.only,
            exclude=args.exclude,
            clean=args.clean,
            quantiles=args.quantiles,
        )

    @classmethod
    def prior_quantile(cls, s: str) -> tuple[str, float]:
        name, quantile = s.split(":")
        try:
            _quantile = float(quantile)
        except ValueError as e:
            raise TypeError(
                f"Can't convert {quantile} to float in ({name}:{quantile})"
            ) from e

        if not (0 <= _quantile <= 1):
            raise ValueError(f"Quantile must be in [0, 1] in ({name}:{_quantile})")

        return name, _quantile

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Generate priors for the benchmarks"
        )
        parser.add_argument("--seed", type=int, default=103_377, help="The seed to use")
        parser.add_argument(
            "--nsamples",
            type=int,
            default=100,
            help="The number of samples to generate",
        )
        parser.add_argument(
            "--prefix", type=str, help="The prefix to use for the generated prior file"
        )
        parser.add_argument(
            "--to", type=Path, default=Path("priors"), help="Where to save the priors"
        )
        parser.add_argument(
            "--fidelity",
            type=float,
            required=False,
            help="The fidelity to evaluated at, defaults to max fidelity",
        )
        parser.add_argument(
            "--quantiles",
            type=self.prior_quantile,
            nargs="+",
            help="The quantiles to use for the priors with their name (name:quantile)",
            default=[
                ("really_good", 0.95),
                ("good", 0.75),
                ("bad", 0.25),
                ("really_bad", 0.1),
            ],
        )
        parser.add_argument(
            "--only",
            type=str,
            nargs="*",
            choices=list(mfpbench._mapping),
            help="Only generate priors for these benchmarks",
        )
        parser.add_argument(
            "--exclude",
            type=str,
            nargs="*",
            choices=list(mfpbench._mapping),
            help="Exclude benchmarks",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Clean out any files in the directory first",
        )
        return parser


def main() -> int:
    """The main entry point."""
    parser = argparse.ArgumentParser()
    handlers = {
        "download": DownloadHandler(),
        "generate-priors": GeneratePriorsHandler(),
    }
    subparsers = parser.add_subparsers(dest="command")
    for name, handle in handlers.items():
        subparsers.add_parser(name, parents=[handle.parser], add_help=False)

    args = parser.parse_args()
    handler = handlers.get(args.command)

    if handler is None:
        raise ValueError(f"Unknown command {args.command}")

    handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
