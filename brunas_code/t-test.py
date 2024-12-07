import numpy as np
from scipy.stats import ttest_rel

# Example data: Replace these with actual performance metrics for each optimizer
# Each list should contain the performance metrics (e.g., accuracy) across seeds
# Adam = np.array([])
# Adamax = np.array([])
# Adadelta = np.array([])
# LBFGS = np.array([0.85, 0.86, 0.84, 0.85, 0.87])

optimizers = {'Adadelta': [0.9818, 0.9784, 0.9818, 0.9816, 0.9792], 'Adam': [0.9717, 0.9675, 0.9622, 0.9696, 0.9686], 'Adamax': [0.9781, 0.9778, 0.9793, 0.978, 0.9769], 'L-BFGS': [0.098, 0.9538, 0.9551, 0.9406, 0.9275]}

# Create a list of optimizer names and their corresponding data
# optimizers = {
#     "Adam": Adam,
#     "Adamax": Adamax,
#     "Adadelta": Adadelta,
#     "LBFGS": LBFGS,
# }

# Perform paired t-tests for all pairs of optimizers
print("Paired t-test Results:")
for opt1, data1 in optimizers.items():
    for opt2, data2 in optimizers.items():
        if opt1 != opt2:
            t_stat, p_value = ttest_rel(data1, data2)

            # print(f"{opt1} vs {opt2}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
            if p_value < 0.05:
                print(f"SIGNIFICANT: {opt1} vs {opt2}")
            else:
                print(f"NOT SIGNIFICANT: {opt1} vs {opt2}")
