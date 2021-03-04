import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt


def convert_score_to_label(scores, is_logit=True, threshold=0.5) -> torch.Tensor:
    if is_logit:
        p = torch.sigmoid(scores)
    else:
        p = scores
    if threshold:
        p[p > threshold] = 1
        p[p < threshold] = 0
        p = p.type(torch.bool)
    return p


def convert_tensor_to_image(x: torch.Tensor) -> np.ndarray:
    x = x.view(-1, 28, 28, 1)
    x = x.cpu().numpy()
    return x


def create_grid_plot(images, labels) -> plt.Figure:
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    images = convert_tensor_to_image(images)[:4 * 4]
    values = convert_score_to_label(labels)

    for idx, image in enumerate(images):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis("off")
        axes[row, col].set_title(str(values[idx].item()), size=10)
        axes[row, col].imshow(image.reshape(28, 28), cmap="gray", aspect="auto")
    return fig
