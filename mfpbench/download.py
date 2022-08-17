from __future__ import annotations

from abc import ABC, abstractmethod

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DATAROOT = Path("data")


@dataclass(frozen=True)  # type: ignore[misc]
class Source(ABC):
    root: Path = DATAROOT

    @abstractmethod
    def download(self) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def path(self) -> Path:
        return self.root / self.name

    def exists(self) -> bool:
        return self.path.exists()


@dataclass(frozen=True)
class YAHPOSource(Source):
    tag: str = "v1.0"
    git_url: str = "https://github.com/slds-lmu/yahpo_data"

    @property
    def cmd(self) -> str:
        return f"git clone --depth 1 --branch {self.tag} {self.git_url} {self.path}"

    @property
    def name(self) -> str:
        return "yahpo-gym-data"

    def download(self) -> None:
        print(f"Downloading to {self.path}")
        print(f"$ {self.cmd}")
        subprocess.run(self.cmd.split())


@dataclass(frozen=True)
class JAHSBenchSource(Source):
    # Should put data version info here

    @property
    def name(self) -> str:
        return "jahs-bench-data"

    @property
    def cmd(self) -> str:
        return f"python -m jahs_bench.download --save_dir {self.path}"

    def download(self) -> None:
        print(f"Downloading to {self.path}")
        print(f"$ {self.cmd}")
        subprocess.run(self.cmd.split())


sources = {source.name: source for source in [YAHPOSource(), JAHSBenchSource()]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    force = args.force

    root = Path(args.data_dir) if args.data_dir is not None else DATAROOT
    root.mkdir(exist_ok=True)

    download_sources = [
        YAHPOSource(root=root),
        JAHSBenchSource(root=root),
    ]

    for source in download_sources:
        if source.exists() and force:
            shutil.rmtree(source.path)

        if not source.exists():
            source.download()
        else:
            print(f"Source already downloaded: {source}")

        if not source.path.exists():
            raise RuntimeError(f"Something went wrong downloading {source}")
