import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

path = "Variances/Combined.csv"
if not os.path.isfile(path):
    path = "Variances/results_R"

    full_res = pd.DataFrame(columns=["Run", "L.R.Power", "initial_accumulator_value", "l1_regularization_strength", "l2_regularization_strength", "Variance"])

    for r in range(5):
        for w in range(5):
            for z in range(5):
                data = np.load(path + str(r) + "_W" + str(w) + "_z" + str(z) + ".npy")[r, :5, :5]
                for x in range(5):
                    for y in range(5):
                        full_res.loc[full_res.shape[0]] = [str(r), str(x/5), str(y/5), str(z/5), str(w/5), data[x, y]]
    full_res.to_csv("Variances/Combined.csv")
else:
    full_res = pd.read_csv("Variances/Combined.csv")

# full_res = full_res[full_res["L.R.Power"] != 0.0]
# full_res = full_res[full_res["L.R.Power"] != 0.2]
full_res = full_res[full_res["L.R.Power"] == 0.8]
full_res = full_res[full_res["initial_accumulator_value"] == 0.2]
full_res = full_res[full_res["l2_regularization_strength"] > 0.2]
full_res = full_res[full_res["l1_regularization_strength"] < 0.4]


sns.boxplot("l2_regularization_strength", "Variance", "l1_regularization_strength", data=full_res)
plt.title("initial_accumulator_value=0.2, L.R.Power=0.8")
plt.show()

