import os

import setuptools

from mfpbench import (
    author,
    author_email,
    description,
    package_name,
    project_urls,
    url,
    version,
)

HERE = os.path.dirname(os.path.realpath(__file__))


def read_file(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as fh:
        return fh.read()


extras_require = {
    "dev": [
        # Test
        "pytest>=4.6",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        "pytest-cases"
        # Docs
        "automl_sphinx_theme",
        # Others
        "isort",
        "black",
        "pydocstyle",
        "pre-commit",
    ]
}

setuptools.setup(
    name=package_name,
    author=author,
    author_email=author_email,
    description=description,
    long_description=read_file(os.path.join(HERE, "README.md")),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url=url,
    project_urls=project_urls,
    version=version,
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=[
        "tqdm",
        "pyyaml",
        "numpy>=1.0.0",
        # We need to hardlock the benchmarks to provide consistency
        "yahpo-gym==1.0.1",
        "jahs_bench @ git+https://github.com/automl/jahs_bench_201.git@880fbcb35a83df7b6c02440a6c13adb921f54657#egg=jahs_bench",  # noqa
        "xgboost>=1.0.0",
    ],
    extras_require=extras_require,
    test_suite="pytest",
    platforms=["Linux"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
