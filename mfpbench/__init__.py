import datetime

from mfpbench.jahs import (
    JAHSCifar10,
    JAHSColorectalHistology,
    JAHSConfig,
    JAHSConfigspace,
    JAHSFashionMNIST,
    JAHSResult,
)

name = "mf-prior-bench"
package_name = "mfpbench"
author = "bobby1 and bobby2"
author_email = "eddiebergmanhs@gmail.com"
description = "No description given"
url = "https://www.automl.org"
project_urls = {
    "Documentation": "https://automl.github.io/mf-prior-bench/main",
    "Source Code": "https://github.com/automl/mfpbench",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, bobby1 and bobby2"
version = "0.0.1"

_mapping = {
    "jahs_cifar_10": JAHSCifar10
}

def get_benchmark(name: str):
    ...

__all__ = [
    "name",
    "package_name",
    "author",
    "author_email",
    "description",
    "url",
    "project_urls",
    "copyright",
    "version",
    "JAHSCifar10",
    "JAHSColorectalHistology",
    "JAHSFashionMNIST",
    "JAHSConfigspace",
    "JAHSConfig",
    "JAHSResult",
]
