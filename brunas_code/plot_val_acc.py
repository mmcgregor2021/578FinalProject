import matplotlib.pyplot as plt
import numpy as np
import os

PATH = "/Users/brunajasinowodolinski/Desktop/project/"

all_val_accuracies = {'Adam': [[95.58333333333333, 95.99166666666666, 95.88333333333334, 96.475, 96.50833333333334, 96.09166666666667, 96.34166666666667, 96.30833333333334, 95.93333333333334, 96.76666666666667, 96.30833333333334, 96.61666666666666, 96.78333333333333, 96.83333333333333, 97.01666666666667, 96.96666666666667, 96.85833333333333, 96.44166666666666, 96.93333333333334, 96.04166666666667]], 'Adadelta': [[95.50833333333334, 96.48333333333333, 97.23333333333333, 97.725, 97.18333333333334, 97.36666666666666, 97.625, 97.475, 97.50833333333334]], 'Adamax': [[95.05833333333334, 95.6, 96.8, 96.90833333333333, 97.175, 97.28333333333333, 97.325, 97.38333333333334, 96.83333333333333, 97.275, 97.325, 97.425, 97.55, 97.49166666666666, 97.50833333333334, 97.425, 97.31666666666666, 97.44166666666666]]}

def pad_and_average(accs):
    max_length = max(len(seed_accs) for seed_accs in accs)  # Find the longest array
    padded_accs = np.full((len(accs), max_length), np.nan)  # Initialize with NaNs

    for i, seed_accs in enumerate(accs):
        padded_accs[i, :len(seed_accs)] = seed_accs  # Fill with seed accs

    mean_accs = np.nanmean(padded_accs, axis=0)  # Compute mean ignoring NaNs
    return mean_accs


# validation accuracies
plt.figure()
for optimizer_type, accuracies in all_val_accuracies.items():
    mean_accuracies = pad_and_average(accuracies)  # Average across seeds
    plt.plot(range(1, len(mean_accuracies) + 1), mean_accuracies, label=optimizer_type)

plt.title(f'Validation Accuracy for Neural Network on MNIST')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(PATH, f'validation_accuracy.png'))
plt.close()
