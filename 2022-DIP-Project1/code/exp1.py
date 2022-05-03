import numpy as np

from utils import readimage, saveimage, showimage, show_histogram


def HistogramEqualization(img: np.ndarray):
    show_histogram(img, title='original histogram')
    bins = np.array([(img == i).sum() for i in range(256)])
    probs = bins / bins.sum()
    eq_bins = 255 * np.cumsum(probs)
    eq_bins = np.round(eq_bins).astype(np.uint8)
    eq_img = np.array([eq_bins[i] for i in img.flatten()]).reshape(*img.shape)
    show_histogram(eq_img, title='histogram after equalization')
    showimage(eq_img)
    saveimage(eq_img, './output_1.png')


if __name__ == '__main__':
    img = readimage('../Q1_image/test_2.jpg')
    HistogramEqualization(img)
