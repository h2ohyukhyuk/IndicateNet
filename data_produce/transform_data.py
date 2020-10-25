from constants import *
import cv2
import numpy as np
import random

class SRT(object):
    def __init__(self, val=False, crop_size=512, scale_min=0.6, scale_max=1.4, rot=25, border=128, border_value=(128, 128, 128)):
        self.val = val
        self.crop_size = crop_size
        self.crop_size_half = crop_size // 2
        self.crop_wh = (self.crop_size, self.crop_size)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rot = rot
        self.border = border
        self.border_value = border_value

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def get_params(self, img):

        h, w = img.shape[:2]
        random_scale = self.crop_size / max(h, w)
        center_x, center_y = w / 2.0, h / 2.0
        random_trans_x, random_trans_y = 0, 0
        random_rot = 0.0

        if not self.val:
            random_scale = random_scale * np.random.choice(np.arange(self.scale_min, self.scale_max, 0.1))
            w_border = self._get_border(self.border, w)
            h_border = self._get_border(self.border, h)
            random_trans_x = (np.random.randint(low=w_border, high=w - w_border) - center_x) * random_scale
            random_trans_y = (np.random.randint(low=h_border, high=h - h_border) - center_y) * random_scale

            random_rot = random.randint(-int(self.rot), int(self.rot))

        M = cv2.getRotationMatrix2D((center_x, center_y), random_rot, random_scale)

        M[0, 2] += self.crop_size_half - center_x - random_trans_x
        M[1, 2] += self.crop_size_half - center_y - random_trans_y

        return M

    def transform_joint(self, T2D, keypoints):
        """
        joint array shape
        joint2d: (n_human, n_joint, 3) ... x, y, visibility
        """
        n_human, n_joint = keypoints.shape[:2]

        keypoints_t = np.copy(keypoints)

        for i in range(n_human):
            for j in range(n_joint):
                x2d, y2d, v = keypoints[i, j]

                if v != COCO_KPT_NO:
                    keypoints_t[i, j, 0] = T2D[0, 0] * x2d + T2D[0, 1] * y2d + T2D[0, 2]
                    keypoints_t[i, j, 1] = T2D[1, 0] * x2d + T2D[1, 1] * y2d + T2D[1, 2]

                    if not (0 <= keypoints_t[i, j, 0] < self.crop_size):  # x coord
                        keypoints_t[i, j, 2] = COCO_KPT_INVISIBLE

                    if not (0 <= keypoints_t[i, j, 1] < self.crop_size):  # y coord
                        keypoints_t[i, j, 2] = COCO_KPT_INVISIBLE

        return keypoints_t

    def __call__(self, img, keypoints, mask):

        M2D = self.get_params(img)

        img_transf = cv2.warpAffine(img, M2D, self.crop_wh, flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=self.border_value)

        mask_transf = cv2.warpAffine(mask, M2D, self.crop_wh, flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if keypoints is None:
            keypoints_transf = None
        else:
            keypoints_transf = self.transform_joint(M2D, keypoints)

        return img_transf, keypoints_transf, mask_transf

class RandomHorizontalFlip(object):
    """Random horizontal flip the image. for COCO
    Args:
        prob (number): the probability to flip.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    @staticmethod
    def hflip(img, kpt2d, kpt3d):
        height, width, _ = img.shape

        img = img[:, ::-1, :]

        for i in range(kpt2d.shape[1]):
            kpt2d[:, i, 0] = width - 1 - kpt2d[:, i, 0]

        kpt3d[:, :, 0] *= -1  # flip x axis

        _from = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        _to = [0, 1, 5, 6, 7, 2, 3, 4, 9, 8]

        kpt2d[:, _to, :] = kpt2d[:, _from, :]
        kpt3d[:, _to, :] = kpt3d[:, _from, :]

        return np.ascontiguousarray(img), kpt2d, kpt3d

    def __call__(self, img, kpt2, kpt3):
        """
        Args:
            img    (numpy.ndarray): Image to be flipped.
            kpt    (list):          Keypoints to be flipped.
            center (list):          Center points to be flipped.
        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return self.hflip(img, kpt2, kpt3)

        return img, kpt2, kpt3


class PixelNoise(object):
    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    @staticmethod
    def get_params(noise_factor):
        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        return pn

    def __call__(self, img, kpt):
        pn = self.get_params(self.noise_factor)

        img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img[:, :, 0] * pn[0]))
        img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img[:, :, 1] * pn[1]))
        img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img[:, :, 2] * pn[2]))

        return img, kpt


class Gamma(object):
    def __init__(self, noise_factor=0.4):
        self.noise_factor = noise_factor

    @staticmethod
    def get_params(gamma_range):
        gamma_shift = random.uniform(-gamma_range, gamma_range)
        invGamma = 1.0 / (1.0 + gamma_shift)
        return invGamma

    def __call__(self, img, kpt):

        invGamma = self.get_params(self.noise_factor)

        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        img = cv2.LUT(img, table)

        return img, kpt


class Blur(object):
    def __init__(self, blur=0.5):
        self.blur = blur

    @staticmethod
    def get_params(blur):
        prob = random.uniform(0.0, 1.0)
        if prob < blur:
            return True

        return False

    def __call__(self, img, kpt):

        boolBlur = self.get_params(self.blur)

        if boolBlur:
            img_w, img_h = img.shape[0], img.shape[1]
            img_size = min(img_w, img_h)
            size_gaussian_blur = random.randint(3, max(3, int(img_size / 15)))
            if size_gaussian_blur % 2 == 0:
                size_gaussian_blur = size_gaussian_blur + 1
            img = cv2.GaussianBlur(img, (size_gaussian_blur, size_gaussian_blur), sigmaX=0)

        return img, kpt


class Normalize(object):

    def __init__(self, mean=128.0, std=256.0):
        self.mean = mean
        self.std = std

    def __call__(self, img, kpt):

        img = (img - self.mean)/self.std

        return img, kpt

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transform.Compose([
        >>>      transform.RandomResized(),
        >>>      transform.RandomHorizontalFlip(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, keypoints, mask):

        for t in self.transforms:
            if isinstance(t, SRT):
                img, keypoints, mask = t(img, keypoints, mask)
            elif isinstance(t, RandomHorizontalFlip):
                img, keypoints = t(img, keypoints)
            else:
                img, keypoints = t(img, keypoints)

        return img, keypoints, mask