from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import json
import os
import cv2
import math

from constants import *

class DatasetCOCO():
    num_joints = 17
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]

    def __init__(self, split, path):
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [4, 6], [3, 5], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [6, 12], [5, 11], [11, 12],
                      [12, 14], [14, 16], [11, 13], [13, 15]]

        self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.data_dir = path
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))

        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'person_keypoints_{}2017.json').format(split)
        self.max_objs = 32
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def get_example(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        crowd_mask = np.zeros((h, w), dtype='bool')
        unannotated_mask = np.zeros((h, w), dtype='bool')
        instance_masks = []
        keypoints = []

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            num_keypoints = ann['num_keypoints']
            pts = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            area = ann['area']

            if area == 0:
                continue

            mask = self.coco.annToMask(ann)

            # if crowd, don't compute loss
            if ann['iscrowd'] == 1:
                crowd_mask = np.logical_or(crowd_mask, mask)
            # if tiny instance, don't compute loss
            elif num_keypoints == 0:
                unannotated_mask = np.logical_or(unannotated_mask, mask)
                instance_masks.append(mask)
            else:
                instance_masks.append(mask)
                keypoints.append(ann['keypoints'])

    def get_image_keypoints_mask(self, index):

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        ignore_mask = np.zeros((h, w), dtype='bool')
        list_kpts = []

        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(ann['category_id']) - 1

            if cls_id != COCO_CLASS_PERSON:
                continue

            if ann['area'] == 0:
                continue

            mask = self.coco.annToMask(ann=ann)

            # if crowd, don't compute loss
            if ann['iscrowd'] == 1:
                ignore_mask = np.logical_or(ignore_mask, mask)
            # if tiny instance, don't compute loss
            elif ann['num_keypoints'] == 0:
                ignore_mask = np.logical_or(ignore_mask, mask)
            else:
                list_kpts.append(ann['keypoints'])

        if len(list_kpts) > 0:
            keypoints = np.array(list_kpts, np.float32).reshape(-1, self.num_joints, 3)
        else:
            keypoints = None

        ignore_mask = ignore_mask.astype(np.uint8)
        return img, keypoints, ignore_mask

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
                        np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__ == '__main__':
    from utils import draw
    path = 'C:/data/coco2017'
    db_coco = DatasetCOCO(split='train', path=path)

    for i in range(db_coco.num_samples):
        img, keypointss, mask = db_coco.get_image_keypoints_mask(i)

        row, col = np.where(mask > 0)
        img[row, col, :] = db_coco.mean*255

        if keypointss is not None:
            print(keypointss.shape)

        for keypoints in keypointss:
            img = draw.draw_keypoints(img, keypoints)

        cv2.imshow('coco', img)
        cv2.waitKey()