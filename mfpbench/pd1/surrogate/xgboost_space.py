from __future__ import annotations

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

# This is used for n_estimators parameter of xgboost
MIN_ESTIMATORS = 50
MAX_ESTIMATORS = 2000


def space(seed: int | None) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)

    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "eta",
                lower=2**-10,
                upper=1.0,
                default_value=0.3,
                log=True,
            ),  # learning rate
            UniformIntegerHyperparameter(
                "max_depth",
                lower=1,
                upper=50,
                default_value=10,
                log=True,
            ),
            UniformFloatHyperparameter(
                "colsample_bytree",
                lower=0.1,
                upper=1.0,
                default_value=1.0,
                log=True,
            ),
            UniformFloatHyperparameter(
                "reg_lambda",
                lower=2**-10,
                upper=2**10,
                default_value=1,
                log=True,
            ),
            UniformFloatHyperparameter(
                "subsample",
                lower=0.1,
                upper=1,
                default_value=1,
                log=False,
            ),
            CategoricalHyperparameter(
                "booster",
                choices=["dart", "gbtree"],
                default_value="gbtree",
            ),
            UniformFloatHyperparameter(
                "alpha",
                lower=1e-4,
                upper=1e4,
                default_value=1,
                log=True,
            ),
        ]
    )
    return cs
