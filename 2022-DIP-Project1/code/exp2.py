import numpy as np

from utils import readimage, saveimage, showimage


def erosion(img: np.ndarray):
    se = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    h, w = img.shape[-2:]
    ero_img = np.zeros(img.shape, dtype=np.uint8)
    pad_img = np.ones((h+2, w+2), dtype=np.uint8)
    pad_img[1:h+1, 1:w+1] = img
    for i in range(1, h+1):
        for j in range(1, w+1):
            if (pad_img[i-1:i+2, j-1:j+2] * se).sum() == 9:
                ero_img[i-1, j-1] = 1
    return ero_img


def BoundaryExtraction(img: np.ndarray):
    ero_img = erosion(img)
    return img - ero_img


def Thresholding(img: np.ndarray, threshold):
    loc = img <= threshold
    binary_img = np.zeros(img.shape, dtype=np.uint8)
    binary_img[loc] = 0
    binary_img[~loc] = 1
    return binary_img


if __name__ == '__main__':
    img = readimage('../Q2_image/test.png')[:, :, 0]
    img = Thresholding(img, threshold=230)
    img = 1 - img
    bound_img = BoundaryExtraction(img)
    bound_img = bound_img * 255
    showimage(bound_img)
    saveimage(bound_img, './output_2.png')
