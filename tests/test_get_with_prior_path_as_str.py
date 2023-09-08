from __future__ import annotations

from pathlib import Path

from pytest_cases import parametrize

import mfpbench

# Just get the first mfhartmann benchmark name as it doesn't need a datadir
FIXTURE_BENCH = next(name for name in list(mfpbench._mapping) if name.startswith("mfh"))


@parametrize("suffix", ["yaml", "yml", "json"])
def test_get_with_valid_file_as_str(tmp_path: Path, suffix: str) -> None:
    bench = mfpbench.get(FIXTURE_BENCH, seed=1)
    config = bench.sample()

    filepath = tmp_path / f"test.{suffix}"
    config.save(filepath)

    str_filepath = str(filepath)
    bench = mfpbench.get(FIXTURE_BENCH, seed=1, prior=str_filepath)

    assert bench.prior == config
    assert bench.space.get_default_configuration() == config
