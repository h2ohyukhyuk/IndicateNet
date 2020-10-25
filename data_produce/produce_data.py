
import configure
from dataset.coco_hp import DatasetCOCO
from data_produce import transform_data

class DataProducer():

    def __init__(self, mode, dic_augmentation):

        self.mode = mode
        self.dic_aug = dic_augmentation

        self.coco = DatasetCOCO(path=configure.PATH_COCO, split=mode)
        self.num_data = self.coco.num_samples

        self.transform = self.build_transform()

    def get_num_data(self):
        return self.num_data

    def build_transform(self):

        srt = transform_data.SRT()
        trasform = transform_data.Compose([srt])
        return trasform

    def get(self, index):

        img, keypoints, mask = self.coco.get_image_keypoints_mask(index=index)

        img_aug, keypoints_aug, mask_aug = self.transform(img, keypoints, mask)

        return img_aug, keypoints_aug, mask_aug


if __name__ == '__main__':
    from utils import draw
    import numpy as np
    import cv2

    dic_augmentation = {}
    dic_augmentation['rot'] = 25
    dp = DataProducer(mode='train', dic_augmentation=dic_augmentation)

    for i in range(dp.get_num_data()):
        img, keypointss, mask = dp.get(i)

        row, col = np.where(mask > 0)
        img[row, col, :] = 128

        if keypointss is not None:
            for keypoints in keypointss:
                img = draw.draw_keypoints(img, keypoints)

        cv2.imshow('coco', img)
        cv2.waitKey()