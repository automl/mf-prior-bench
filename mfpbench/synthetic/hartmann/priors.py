from mfpbench.synthetic.hartmann.config import MFHartmann3Config, MFHartmann6Config
from mfpbench.synthetic.hartmann.generators import MFHartmann3, MFHartmann6

HARTMANN3D_PRIORS = {
    "good": MFHartmann3Config(
        X_0=0.5804173195470849,
        X_1=0.49455702088463216,
        X_2=0.8246093619057706,
    ),
    "bad": MFHartmann3Config(
        X_0=0.8164723244941313,
        X_1=0.5588626087033245,
        X_2=0.20921791132268686,
    ),
    "default": MFHartmann3Config(X_0=0.5, X_1=0.5, X_2=0.5),
    "perfect": MFHartmann3Config.from_dict(
        {f"X_{i}": x for i, x in enumerate(MFHartmann3.optimum)}
    ),
}


HARTMANN6D_PRIORS = {
    "good": MFHartmann6Config(
        X_0=0.28836121940052994,
        X_1=0.41544134405398103,
        X_2=0.33149043526417676,
        X_3=0.366326694741957,
        X_4=0.31219533791739007,
        X_5=0.5804458426229278,
    ),
    "bad": MFHartmann6Config(
        X_0=0.6363492772350455,
        X_1=0.02594920126387934,
        X_2=0.1566788901926598,
        X_3=0.13291091536247845,
        X_4=0.878415924184874,
        X_5=0.7098124347339022,
    ),
    "default": MFHartmann6Config(
        X_0=0.5,
        X_1=0.5,
        X_2=0.5,
        X_3=0.5,
        X_4=0.5,
        X_5=0.5,
    ),
    "perfect": MFHartmann6Config.from_dict(
        {f"X_{i}": x for i, x in enumerate(MFHartmann6.optimum)}
    ),
}
