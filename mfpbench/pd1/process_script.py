from pathlib import Path


def process(datadir: Path) -> None:
    if not datadir.exists():
        raise FileNotFoundError(
            f"Can't find folder at {datadir}, have you run\n"
            f"`python -m mfpbench.download --data-dir {datadir.parent}`"
        )


if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().absolute().parent
    DATADIR = HERE.parent.parent / "data" / "pd1-data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    datadir = args.data_dir if args.data_dir else DATADIR
    process(datadir)
