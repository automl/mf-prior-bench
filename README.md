# mf-prior-bench

## Installation
```bash
git clone git@github.com:automl/mf-prior-bench.git
cd mf-prior-bench

# Create your env however
...

# I use pip and virtualenv
pip install -e  ".[dev]"

# Pre-commit
pre-commit install

# Optionally there's a makefile command for that
make install-dev  # use `make help` if you like

# You're going to want to download things needed jahs-bench and YAHPO
# Just leave --data-dir at default to be safest  "./data/{jahs-bench-data, yahpo-gym-data}"
python -m mfpbench.download [--force] [--data-dir PATH]
```

### For early stages
Scaffolding around jahs-bench has been set up and you can mostly just refer to the code within this
repo to know exactly the info you can provide and get. Check out the `example1.py` to get an overview of how
to interact. Use your code editor on the types provided to get a good idea of whats there.

Check out `mfpbench/jahs/{benchmark,config,result}.py` if you like specifics otheriwse `mfpbench/{benchmark,config,result}.py` for the abstract. Most documentation is on the latter but hopefully there's enough info from types to know what's going on.
