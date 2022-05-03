import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def readimage(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.array(img)  # noqa
    return img


def saveimage(img: np.ndarray, path: str) -> None:
    img = Image.fromarray(img)
    img.save(path)
    print(f'Image is successfully saved to {path}.')


def showimage(img: np.ndarray) -> None:
    img = Image.fromarray(img)
    img.show()


def show_histogram(img: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 1)
    ax.hist(img.flatten(), bins=np.arange(256))
    ax.set_title(title)
    plt.show()
    plt.close(fig)
