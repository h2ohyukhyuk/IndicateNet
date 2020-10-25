
from constants import *
import cv2

def draw_keypoints(img, keypoints, radius=3, line_thick=-1):
    img = img.copy()

    if keypoints is None:
        return img

    if keypoints.shape[1] == 3:
        for k, kpt in enumerate(keypoints):
            x, y, v = kpt
            pt = (int(x), int(y))
            if v == COCO_KPT_VISIBLE:
                cv2.circle(img, pt, int(radius), colors.get(k), line_thick)
            elif v == COCO_KPT_INVISIBLE:
                cv2.circle(img, pt, int(radius+2), (128, 128, 128), line_thick)
                cv2.circle(img, pt, int(radius), colors.get(k), line_thick)
    elif keypoints.shape[1] == 2:
        for k, kpt in enumerate(keypoints):
            x, y = kpt
            pt = (int(x), int(y))
            cv2.circle(img, pt, int(radius), colors.get(k), line_thick)

    return img