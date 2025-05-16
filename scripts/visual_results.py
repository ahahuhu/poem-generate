import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List

FILE_PATH = [
    'result/30-0.0001-peft.txt',
    'result/20-0.01.txt',
    'result/20-0.001.txt',
    'result/20-0.0001.txt',
]

def visual_result(file_paths: List[str], save_path: str = 'result/result.svg'):
    fig, ax = plt.subplots(figsize=(8, 6))
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue
        with open(file_path) as f:
            data = f.readlines()
        # Skip lines that can't be parsed, and handle empty/short files gracefully
        losses = []
        for index in range(4, len(data), 2):
            try:
                losses.append(float(data[index].split()[-1]))
            except (IndexError, ValueError):
                continue
        if not losses:
            print(f"Warning: No valid loss data in {file_path}")
            continue
        epochs = np.arange(1, len(losses)+1)
        label = os.path.splitext(os.path.basename(file_path))[0]
        ax.plot(epochs, losses, label=label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    visual_result(FILE_PATH)