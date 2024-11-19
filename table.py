import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Placeholder data - Replace with your actual results
optimizers = ["Adam", "Adamax", "Adadelta", "L-BFGS"]
adam = [0.9232, 0.9232, 0.9145,0.9194 , 0.9173]
adamax = [0.9251, 0.9251, 0.9245, 0.9265, 0.9244]
lbfgs = [0.9129, 0.90960, 0.9111, 0.9063, 0.9061]
adadelta = [0.8834, 0.8835, 0.8836, 0.8837, 0.8803]
            

accuracies = [np.mean(adam), np.mean(adamax), np.mean(adadelta), np.mean(lbfgs)]
stds = [np.std(adam), np.std(adamax), np.std(adadelta), np.std(lbfgs)]
print(accuracies)
# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(optimizers, accuracies, color="cornflowerblue", edgecolor="black")

# Add value labels on top of each bar, including mean ± std
for bar, acc, std in zip(bars, accuracies, stds):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,  # Position above the bar
        f"{acc * 100:.2f} ± {std * 100:.2f}%",  # Format with mean ± std
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Add labels and title
plt.xlabel("Optimizers", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Final Test Accuracy for Different Optimizers on MNIST, Logistic Regression", fontsize=14)

# Set the y-axis limits for better readability
plt.ylim(0.8, 0.96)  # Adjust limits for accuracy values

# Customize grid lines
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display the chart
plt.tight_layout()
plt.show()
