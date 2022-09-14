from __future__ import annotations

from abc import ABC, abstractmethod

import argparse
import gzip
import json
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


@dataclass(frozen=True)
class PD1Source(Source):

    url: str = "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
    unpacked_folder_name: str = "pd1"

    @property
    def name(self) -> str:
        return "pd1-data"

    @staticmethod
    def unpack_jsonl(path: Path) -> list[dict]:
        assert path.name.endswith(".jsonl.gz"), path

        with gzip.open(path, mode="rt") as f:
            data = [json.loads(line) for line in f]

        return data

    def download(self) -> None:
        import urllib.request

        tarpath = self.path / "data.tar.gz"

        # Download the file
        print(f"Downloading from {self.url} to {tarpath}")
        with urllib.request.urlopen(self.url) as response, open(tarpath, "wb") as f:
            shutil.copyfileobj(response, f)

        # Unpack it to the datadir
        print(f"Unpacking {tarpath}")
        shutil.unpack_archive(tarpath, self.path)

        unpacked_folder = self.path / self.unpacked_folder_name

        print(f"Moving files from {unpacked_folder} to {self.path}")
        for filepath in unpacked_folder.iterdir():
            to = self.path / filepath.name
            print(f"Move {filepath} to {to}")
            shutil.move(str(filepath), str(to))

        # Unpacking gzipped json files
        print("Unpacking gzipped json.gz files")
        files = [f for f in self.path.iterdir() if f.name.endswith("jsonl.gz")]
        for file in files:
            new_path = file.parent / file.name.replace(".jsonl.gz", ".json")

            print(f"Processing {file} to {new_path}")
            data = self.unpack_jsonl(file)
            with open(new_path, mode="w") as out:
                json.dump(data, out)

        shutil.rmtree(str(unpacked_folder))


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
        PD1Source(root=root),
    ]

    for source in download_sources:
        if source.exists() and force:
            shutil.rmtree(source.path)

        if not source.exists():
            source.path.mkdir(exist_ok=True)
            source.download()
        else:
            print(f"Source already downloaded: {source}")

        if not source.path.exists():
            raise RuntimeError(f"Something went wrong downloading {source}")
