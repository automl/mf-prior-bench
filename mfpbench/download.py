from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from abc import ABC, abstractmethod
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
        subprocess.run(self.cmd.split())


@dataclass(frozen=True)
class PD1Source(Source):
    url: str = "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
    surrogate_url: str = (
        "https://ml.informatik.uni-freiburg.de/research-artifacts/mfp-bench"
    )
    surrogate_version: str = "vPaper-PriorBand"

    @property
    def name(self) -> str:
        return "pd1-data"

    def download(self) -> None:
        self.download_rawdata()
        self.download_surrogates()

    def download_rawdata(self) -> None:
        tarpath = self.path / "data.tar.gz"

        # Download the file
        with urllib.request.urlopen(self.url) as response, open(tarpath, "wb") as f:
            shutil.copyfileobj(response, f)

        # We offload to a special file for doing all the processing of pd1 into datasets
        from mfpbench.pd1.processing.process_script import process_pd1

        process_pd1(tarball=tarpath)

    def download_surrogates(self) -> None:
        surrogate_dir = self.path / "surrogates"
        surrogate_dir.mkdir(exist_ok=True, parents=True)
        zip_path = surrogate_dir / "surrogates.zip"

        # Download the surrogates zip
        url = f"{self.surrogate_url}/{self.surrogate_version}/surrogates.zip"
        with urllib.request.urlopen(url) as response, open(zip_path, "wb") as f:
            shutil.copyfileobj(response, f)

        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall(surrogate_dir)


sources = {source.name: source for source in [YAHPOSource(), JAHSBenchSource()]}


def download(datadir: Path | None = None, *, force: bool = False) -> None:
    root = datadir if datadir is not None else DATAROOT
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
            pass

        if not source.path.exists():
            raise RuntimeError(f"Something went wrong downloading {source}")
