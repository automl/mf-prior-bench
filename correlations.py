from itertools import chain, product

import matplotlib.pyplot as plt
import pandas as pd

import mfpbench

benches = {
    "mfh3_good": tuple(),
    "mfh3_moderate": tuple(),
    "mfh3_bad": tuple(),
    "mfh3_terrible": tuple(),
}


results = {"name": [], "itrs": [], "spearman": [], "kendalltau": [], "cosine": []}
for (name, args), i in product(benches.items(), [10, 100, 1000]):
    bench = mfpbench.get(name, *args)

    configs = bench.sample(i)
    start, end, _ = bench.fidelity_range

    results_start = [bench.query(c, at=start) for c in configs]
    results_end = [bench.query(c, at=end) for c in configs]

    rf = bench.frame()
    for result in chain(results_start, results_end):
        rf.add(result)


    # Get out the correlations
    print("COSINE")
    corr_cosine = rf.correlations(at=[start, end], method="cosine")
    corr_cosine = corr_cosine[0, 1]
    results["cosine"].append(corr_cosine)

    print("kendall")
    corr_kendall = rf.correlations(at=[start, end], method="kendalltau")
    corr_kendall = corr_kendall[0, 1]
    results["kendalltau"].append(corr_kendall)

    print("spearman")
    corr_spearman = rf.correlations(at=[start, end], method="spearman")
    corr_spearman = corr_spearman[0, 1]
    results["spearman"].append(corr_spearman)

    results["name"].append(name)
    results["itrs"].append(i)



#from pprint import pprint
#pprint(results)
df = pd.DataFrame.from_dict(results)
print(df)
df.plot()

plt.show()
