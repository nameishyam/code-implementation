import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def show_image_comparison(lr, sr, hr, title=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr); axs[0].set_title("Low-Res"); axs[0].axis("off")
    axs[1].imshow(sr); axs[1].set_title("Super-Res"); axs[1].axis("off")
    axs[2].imshow(hr); axs[2].set_title("High-Res"); axs[2].axis("off")
    if title: plt.suptitle(title)
    plt.tight_layout(); plt.show()

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.show()
