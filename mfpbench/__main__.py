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
        parser.add_argument(
            "--data-dir",
            default=mfpbench.download.DATAROOT,
            type=Path,
            help="Where to save the data",
        )
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
            prior_spec=args.priors,
            clean=args.clean,
            use_hartmann_optimum=args.use_hartmann_optimum,
        )

    @classmethod
    def parse_prior(cls, s: str) -> tuple[str, int, float | None, float | None]:
        name, index, noise, categorical_swap_chance = s.split(":")
        try:
            _index = int(index)
        except ValueError as e:
            raise ValueError(f"Invalid index {index}") from e

        if noise in ("None", "0", "0.0", "0.00"):
            _noise = None
        else:
            try:
                _noise = float(noise)
                if not (0 <= _noise <= 1):
                    raise ValueError(f"noise must be in [0, 1] in ({name}:{_noise})")
            except ValueError as e:
                raise TypeError(
                    f"Can't convert {noise} to float in ({name}:{noise})"
                ) from e

        if categorical_swap_chance in ("None", "0", "0.0", "0.00"):
            _categorical_swap_chance = None
        else:
            try:
                _categorical_swap_chance = float(categorical_swap_chance)
                if not (0 <= _categorical_swap_chance <= 1):
                    raise ValueError(
                        f"categorical_swap_chance must be in [0, 1] in ({s})"
                    )
            except ValueError as e:
                raise TypeError(
                    f"Can't convert categorical_swap_chance ({categorical_swap_chance})"
                    f" to float in ({s})"
                ) from e

        return name, _index, _noise, _categorical_swap_chance

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Generate priors for the benchmarks"
        )
        parser.add_argument("--seed", type=int, default=133_077, help="The seed to use")
        parser.add_argument(
            "--nsamples",
            type=int,
            default=1_000_000,
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
            "--priors",
            type=self.parse_prior,
            nargs="+",
            help=(
                "The <priorname>:<index>:<std>:<categorical_swap_chance>"
                " of the priors to generate. You can use python's negative"
                " indexing to index from the end. If a value for std or"
                " categorical_swap_chance is 0 or None, then it will not"
                " be used. However it must be specified."
            ),
            default=[
                ("good", 0, 0.01, None),
                ("medium", 0, 0.125, None),
                ("bad", -1, None, None)
            ],
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
            "--use-hartmann-optimum",
            type=str,
            nargs="+",
            required=False,
            help=(
                "The name of the prior(s) to replace with the optimum if using"
                " hartmann benchmarks. Must be contained in `--priors`"
            )
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
