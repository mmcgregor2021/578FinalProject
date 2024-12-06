import matplotlib.pyplot as plt
import numpy as np
import os

PATH = "/Users/brunajasinowodolinski/Desktop/578FinalProject/brunas_code"

all_val_accuracies = # add val accuracies here

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
