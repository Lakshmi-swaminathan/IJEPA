import os

import matplotlib.pyplot as plt

def plot_training_history(history, output_path):
    epochs = range(1, len(history.history["loss"]) + 1)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training and validation loss
    axs[0].plot(epochs, history.history["loss"], label="Training Loss")
    axs[0].plot(epochs, history.history["val_loss"], label="Validation Loss")
    axs[0].set_title("Loss Over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot training and validation accuracy
    axs[1].plot(epochs, history.history["accuracy"], label="Training Accuracy")
    axs[1].plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    axs[1].set_title("Accuracy Over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(output_path, "lp_training_metrics.png")
    plt.savefig(plot_path)
    plt.close()