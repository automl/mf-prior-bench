# Setup

## Installation

### Using `pip`

```bash
pip install mfpbench
```

To install specific benchmark dependancies, you can specify any or all of the following:

```bash
pip install "mfpbench[yahpo, jahs-bench, pd1]"
```

### From source

```bash
git clone git@github.com:automl/mf-prior-bench

pip install "mf-prior-bench"
```

To install specific benchmark dependancies, you can specify any or all of the following:

```bash
git clone git@github.com:automl/mf-prior-bench

pip install "mf-prior-bench[yahpo, jahs-bench, pd1]"
```

## CLI
`mpfbench` provides some helpful commands for downloading data as well as generating priors
for benchmarks.

```bash exec="true" source="material-block" result="ansi" title="CLI Help"
python -m mfpbench --help
```

---

### Download
Some benchmark require raw files to work, you can download these using the
`python -m mfpbench download` command.

You can specify the location of the root data directory with `--data-dir`.
We recommend leaving this to the default to make working with benchmarks easier.

=== "`--benchmark`"

    ```bash
    # Disabled in docs
    python -m mfpbench download --benchmark "pd1"
    ```

=== "`--list`"

    ```bash exec="true" source="material-block" result="ansi" title="Download Status"
    python -m mfpbench download --list
    ```

=== "`--status`"

    ```bash exec="true" source="material-block" result="ansi" title="Setup Status"
    python -m mfpbench download --status
    ```

=== "`--force`"

    ```bash
    # Execution disabled for docs
    python -m mfpbench download --benchmark "pd1" --force
    ```

=== "`--help`"

    ```bash exec="true" source="material-block" result="ansi" title="Download Help"
    python -m mfpbench download --help
    ```

---

### Install
As there will be conflicting dependancies between different benchmarks, we added some requirements
files to specifically run these benchmarks. You will still have to setup an environment however you
would do so but you can quickly install the required dependancies using the `python -m mfpbench install`
subcommand.

=== "`--benchmark`"

    ```bash
    # Disabled in docs
    python -m mfpbench install --benchmark "lcbench-tabular"
    ```

=== "`--list`"

    ```bash exec="true" source="material-block" result="ansi" title="Install list"
    python -m mfpbench install --list
    ```

=== "`--view`"

    ```bash exec="true" source="material-block" result="ansi" title="Install view"
    python -m mfpbench install --view --benchmark "yahpo"
    ```

=== "`--requirements`"

    ```bash
    # Disabled for docs
    python -m mfpbench install --requirements "/path/to/myreqs"
    ```

=== "`-help`"

    ```bash exec="true" source="material-block" result="ansi" title="Install help"
    python -m mfpbench install --help
    ```

### Generating Priors
TODO
