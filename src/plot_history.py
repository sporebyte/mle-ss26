# src/plot_history.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"


def plot_history(
    history_path: str = OUTPUTS / "history.json",
    out_path: str = OUTPUTS / "loss_curve.png",
):
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, h["train_loss"], label="Train", marker="o")
    plt.plot(epochs, h["val_loss"], label="Validation", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mark the best epoch (lowest val loss)
    best_epoch = h["val_loss"].index(min(h["val_loss"])) + 1
    best_loss = min(h["val_loss"])
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    plt.annotate(
        f"best: epoch {best_epoch}\nval={best_loss:.3f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 1, best_loss + 0.05),
        fontsize=9,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_history()