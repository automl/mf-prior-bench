from __future__ import annotations

from mfpbench.jahs.config import JAHSConfig

DEFAULT_PRIOR = JAHSConfig(
    Activation="ReLU",
    LearningRate=0.1,
    N=5,
    Op1=0,
    Op2=0,
    Op3=0,
    Op4=0,
    Op5=0,
    Op6=0,
    Optimizer="SGD",
    Resolution=1.0,
    TrivialAugment=False,
    W=16,
    WeightDecay=0.0005,
)

CIFAR10_PRIORS = {
    "good": JAHSConfig(
        Activation="Hardswish",
        LearningRate=0.9110690405832061,
        N=5,
        Op1=0,
        Op2=3,
        Op3=2,
        Op4=3,
        Op5=2,
        Op6=2,
        Optimizer="SGD",
        Resolution=1.0,
        TrivialAugment=True,
        W=16,
        WeightDecay=5.172497667031624e-05,
    ),
    "bad": JAHSConfig(
        Activation="Hardswish",
        LearningRate=0.0021820022044816817,
        N=5,
        Op1=1,
        Op2=3,
        Op3=1,
        Op4=2,
        Op5=3,
        Op6=1,
        Optimizer="SGD",
        Resolution=0.5,
        TrivialAugment=False,
        W=16,
        WeightDecay=0.008513658749621622,
    ),
    "default": DEFAULT_PRIOR
}

COLORECTAL_HISTOLOGY_PRIORS = {"default": DEFAULT_PRIOR}
FASHION_MNIST_PRIORS = {"default": DEFAULT_PRIOR}
