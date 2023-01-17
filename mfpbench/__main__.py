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
        mfpbench.download.download(
            datadir=args.data_dir, force=args.force, pd1_dev=args.pd1_dev
        )

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force download and remove existing data",
        )
        parser.add_argument("--data-dir", type=str, help="Where to save the data")
        parser.add_argument(
            "--only",
            type=str,
            nargs="*",
            choices=list(mfpbench.download.sources.keys()),
            help="Only download these",
        )
        parser.add_argument(
            "--pd1-dev", action="store_true", help="Download the pd1 dev data"
        )
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
            hartmann_perfect=args.hartmann_perfect,
            hartmann_optimum_with_noise=args.hartmann_perfect_with_noise,
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

    @classmethod
    def hartmann_priors_noisy(cls, s: str) -> tuple[str, float]:
        name, noise = s.split(":")
        try:
            _noise = float(noise)
            return name, _noise
        except ValueError as e:
            raise TypeError(
                f"Can't convert {noise} to float in ({name}:{noise})"
            ) from e

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
            default=[("good", 0.1), ("bad", 0.9)],
        )
        parser.add_argument(
            "--only",
            type=str,
            nargs="*",
            help="Only generate priors for these benchmarks",
        )
        parser.add_argument(
            "--exclude",
            type=str,
            nargs="*",
            help="Exclude benchmarks",
        )
        parser.add_argument(
            "--clean",
            action="store_true",
            help="Clean out any files in the directory first",
        )
        parser.add_argument(
            "--hartmann-perfect-with-noise",
            nargs="+",
            type=self.hartmann_priors_noisy,
            help=(
                "The (name:noise) of priors to generate from"
                " the Hartmann benchmark's optimum"
            ),
            default=[("perfect-noisy0.25", 0.250)],
        )
        parser.add_argument(
            "--hartmann-perfect",
            action="store_true",
            help="Generate optimum of Hartmann benchmarks",
            default=True,
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
