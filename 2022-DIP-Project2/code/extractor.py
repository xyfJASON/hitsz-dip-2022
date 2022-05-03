import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
from typing import Tuple
from scipy import ndimage as ndi


class BaseExtractor:
    def __init__(self):
        pass

    def extract(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HOGExtractor(BaseExtractor):
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        super().__init__()
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract(self, X: np.ndarray) -> np.ndarray:
        features = [self.compute_feats(x) for x in X]
        features = np.stack(features, axis=0)
        return features

    def compute_feats(self, x: np.ndarray, visualize: bool = False):
        return hog(x, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                   channel_axis=2, visualize=visualize)


class GaborExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    self.kernels.append(kernel)

    def extract(self, X: np.ndarray) -> np.ndarray:
        features_all = [np.concatenate([self.compute_feats(x[:, :, i]) for i in range(3)]) for x in X]
        features_all = np.stack(features_all, axis=0)
        return features_all

    def compute_feats(self, x: np.ndarray):
        feats = np.zeros((len(self.kernels), 2), dtype=np.double)
        for k, kernel in enumerate(self.kernels):
            filtered = ndi.convolve(x, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        feats = feats.flatten()
        return feats


class LBPExtractor(BaseExtractor):
    def __init__(self, numPoints=8, radius=1, method='uniform', pixels_per_block=(32, 32), n_bins=10):
        super().__init__()
        self.numPoints = numPoints
        self.radius = radius
        self.method = method
        self.pixels_per_block = pixels_per_block
        self.n_bins = n_bins

    def extract(self, X: np.ndarray) -> np.ndarray:
        features = [self.compute_feats(x) for x in X]
        features = np.stack(features, axis=0)
        return features

    def compute_feats(self, x: np.ndarray, visualize: bool = False):
        lbp = [local_binary_pattern(x[:, :, i], P=self.numPoints, R=self.radius, method=self.method) for i in range(3)]
        feat = [blocked_histogram(lbp[i], pixels_per_block=self.pixels_per_block, n_bins=self.n_bins) for i in range(3)]
        feat = np.concatenate(feat)
        if visualize:
            return feat, [lbp[i] / lbp[i].max() * 255 for i in range(3)]
        else:
            return feat


def blocked_histogram(feature: np.ndarray, pixels_per_block: Tuple[int, int], n_bins: int, eps=1e-7):
    n_blk_row = (feature.shape[0] + pixels_per_block[0] - 1) // pixels_per_block[0]
    n_blk_col = (feature.shape[1] + pixels_per_block[1] - 1) // pixels_per_block[1]
    feat = []
    for i in range(n_blk_row):
        for j in range(n_blk_col):
            xblock = feature[i*pixels_per_block[0]:min((i+1)*pixels_per_block[0], feature.shape[0]),
                             j*pixels_per_block[1]:min((j+1)*pixels_per_block[1], feature.shape[1])]
            hist, bins = np.histogram(xblock.ravel(), bins=n_bins)
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            feat.append(hist)
    feat = np.concatenate(feat)
    return feat


# another Gabor: no better than the above one
# class GaborExtractor(BaseExtractor):
#     def __init__(self):
#         self.kernels = []
#         ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度，6个
#         # ksize = [13]  # gabor尺度，6个
#         lamda = np.pi / 2.0  # 波长
#         for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
#             for K in range(len(ksize)):
#                 kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
#                 kern /= 1.5 * kern.sum()
#                 self.kernels.append(kern)
#
#
#
#     def extract(self, X: np.ndarray, rgb_feature = False) -> np.ndarray:
#         features_all = []
#         for x in X:
#             if rgb_feature:
#                 features_single = []
#                 for i in range(3):
#                     features_single.append(self.compute_feats(x[:, :, i], self.kernels))
#                 features_single = np.concatenate(features_single, axis=0)
#             else:
#                 x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
#                 features_single = self.compute_feats(x, self.kernels)
#
#             features_all.append(features_single)
#         features_all = np.stack(features_all)
#         return features_all
#
#
#
#     def compute_feats(self, image, kernels):
#         res = []  # 滤波结果
#         for i in range(len(kernels)):
#             # res1 = process(img, filters[i])
#             accum = np.zeros_like(image)
#             for kern in kernels[i]:
#                 fimg = cv2.filter2D(image, cv2.CV_8UC1, kern)
#                 accum = np.maximum(accum, fimg, accum)
#             # accum = np.max(accum, axis=1)
#             res.append(np.asarray(accum).reshape(-1, ))
#         res = np.concatenate(res, axis = 0)
#         return res


def _test_hog():
    from PIL import Image
    import matplotlib.pyplot as plt
    extractor = HOGExtractor()
    img = Image.open('../dataset/train/0_30.jpg')
    img_array = np.array(img)  # noqa
    feature, vis = extractor.compute_feats(img_array, True)
    print(img_array.shape)
    print(feature.shape)
    vis = Image.fromarray(vis)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(vis)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set_title('original')
    ax[1].set_title('hog')
    plt.show()


def _test_lbp():
    from PIL import Image
    import matplotlib.pyplot as plt
    extractor = LBPExtractor()
    img = Image.open('../dataset/train/0_30.jpg')
    img_array = np.array(img)  # noqa
    feature, vis = extractor.compute_feats(img_array, True)
    print(img_array.shape)
    print(feature.shape)
    vis = [Image.fromarray(vis[i]) for i in range(3)]
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[1].imshow(vis[0])
    ax[2].imshow(vis[1])
    ax[3].imshow(vis[2])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    ax[0].set_title('original')
    ax[1].set_title('lbp-r')
    ax[2].set_title('lbp-g')
    ax[3].set_title('lbp-b')
    plt.show()


if __name__ == '__main__':
    _test_hog()
    # _test_lbp()
