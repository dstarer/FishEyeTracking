from opts import Opt
import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from datetime import datetime as dt
import glob
import cv2
import numpy as np


class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def get_left_intrinsics(self):
        raise NotImplementedError()

    def get_right_intrinsics(self):
        raise NotImplementedError()


class RobotCarDataset(Dataset):
    def __init__(self, opts):
        self.left_images, self.left_detections, self.right_images, self.right_detections = self._read_metas_(opts)
        self.root = opts.dir

        self.left_k = np.array([[400.0, 0, 500.107605], [0, 400.0, 511.461426], [0, 0, 1]], dtype=np.float32)
        self.right_k = np.array([[400.0, 0, 502.503754], [0, 400.0, 490.259033], [0, 0, 1]], dtype=np.float32)

    def __load_camera(self, timestamps_path, detection_path):
        imgs = []
        detections = {}
        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file %s" % timestamps_path)
        with open(timestamps_path, "r") as f:
            for line in f.readlines():
                tokens = line.split()
                datetime = dt.utcfromtimestamp(int(tokens[0]) / 1000000)
                filename = tokens[0] + '.jpg'
                imgs.append((datetime, filename))

        if not os.path.isfile(detection_path):
            raise IOError("Could not find detection file %s" % detection_path)
        is_header = True
        with open(detection_path, "r") as f:
            for line in f.readlines():
                if is_header:
                    is_header = False
                    continue
                tokens = line.split(",")
                img = tokens[0].split('/')[-1]
                idx = int(tokens[1])
                cls = int(tokens[2])
                tlx, tly, brx, bry, score = map(float, tokens[3:])
                detections.setdefault(img, [])
                detections[img].append({"id": idx, 'cls': cls, 'box': [tlx, tly, brx, bry], 'score': score})
        return imgs, detections

    def _read_metas_(self, opts):
        timestamp_path = os.path.join(opts.dir, "mono_left.timestamps")
        detection_path = os.path.join(opts.dir, "mono_left_2d_detection.csv")
        left_images, left_detections = self.__load_camera(timestamp_path, detection_path)

        timestamp_path = os.path.join(opts.dir, 'mono_right.timestamps')
        detection_path = os.path.join(opts.dir, "mono_right_2d_detection.csv")
        right_images, right_detections = self.__load_camera(timestamp_path, detection_path)

        return left_images, left_detections, right_images, right_detections

    def __len__(self):
        return min(len(self.left_images), len(self.right_images))

    def __getitem__(self, item):
        left_img_time, left_img_id = self.left_images[item]
        right_img_time, right_img_id = self.right_images[item]
        left_img_path = os.path.join(self.root, "mono_left_undistort", left_img_id)
        right_img_path = os.path.join(self.root, "mono_right_undistort", right_img_id)

        left_image = cv2.imread(left_img_path)
        right_image = cv2.imread(right_img_path)
        left_detection = self.left_detections[left_img_id]
        right_detection = self.right_detections[right_img_id]
        result = {"left_image": left_image, "right_image": right_image,
                  "left_detection": left_detection, "right_detection": right_detection}
        return result

    def get_left_intrinsics(self):
        return self.left_k

    def get_right_intrinsics(self):
        return self.right_k


class PlusAIDataSet(Dataset):
    LEFT_SIDE_CAMERA = "side_left_camera"
    RIGHT_SIDE_CAMERA = "side_right_camera"

    def __init__(self, opts):

        self._bag_path = opts.dir
        if opts.fov == 120:
            self._left_side_k = self._load_intrinsics(os.path.join(opts.calib_path, "left_FOV_120_calibration.txt"))
            self._right_side_k = self._load_intrinsics(os.path.join(opts.calib_path, "right_FOV120_calibration.txt"))
        elif opts.fov == 150:
            self._left_side_k = self._load_intrinsics(os.path.join(opts.calib_path, "FOV_150_left_side_calibration.txt"))
            self._right_side_k = self._load_intrinsics(os.path.join(opts.calib_path, "FOV_150_right_side_calibration.txt"))

        print ("left_side calibration: ", self._left_side_k)
        self._left_imgs = self._scan_imgs(os.path.join(opts.dir, self.LEFT_SIDE_CAMERA))
        self._right_imgs = self._scan_imgs(os.path.join(opts.dir, self.RIGHT_SIDE_CAMERA))
        self._side_left_detections = self._read_detections(
            os.path.join(opts.dir, self.LEFT_SIDE_CAMERA + "_detection.csv"))
        self._side_right_detections = self._read_detections(
            os.path.join(opts.dir, self.RIGHT_SIDE_CAMERA + "_detection.csv"))

    @staticmethod
    def _load_intrinsics(calib_file):
        calib = np.identity(3, dtype=np.float32)
        with open(calib_file, "r") as f:
            for line in f.readlines():
                if line.startswith("calib:"):
                    values = map(float, [word.strip() for word in line.strip().split(":")[1].split(",")])
                    calib[0, 0] = values[0]
                    calib[1, 1] = values[1]
                    calib[0, 2] = values[2]
                    calib[1, 2] = values[3]
        return calib

    @staticmethod
    def _scan_imgs(bag_path):
        print bag_path
        if not os.path.exists(bag_path):
            return []
        image_names = [name.split("/")[-1] for name in glob.glob(bag_path + "/*.jpg")]
        return sorted(image_names)

    @staticmethod
    def _read_detections(detection_path, is_header=True):
        if not os.path.exists(detection_path):
            return {}
        detections = {}
        with open(detection_path, "r") as f:
            for line in f.readlines():
                if is_header:
                    is_header = False
                    continue
                tokens = line.split(",")
                img = tokens[0].split('/')[-1]
                idx = int(tokens[1])
                cls = int(tokens[2])
                tlx, tly, brx, bry, score = map(float, tokens[3:])
                detections.setdefault(img, [])
                detections[img].append({"id": idx, 'cls': cls, 'box': [tlx, tly, brx, bry], 'score': score})
        return detections

    def __getitem__(self, item):
        left_img_id = self._left_imgs[item]
        # right_img_id = self._right_imgs[item]

        left_img_path = os.path.join(self._bag_path, self.LEFT_SIDE_CAMERA, left_img_id)
        # right_img_path = os.path.join(self._bag_path, self.RIGHT_SIDE_CAMERA, right_img_id)

        left_detection = self._side_left_detections.get(left_img_id)
        # right_detection = self._side_right_detections.get(right_img_id)

        left_image = cv2.imread(left_img_path)
        # right_image = cv2.imread(right_img_path)

        # result = {"left_image": left_image, "left_detection": left_detection,
        #           "right_image": right_image, "right_detection": right_detection}
        result = {"left_image": left_image, "left_detection": left_detection}
        return result

    def __len__(self):
        # return min(len(self._left_imgs), len(self._right_imgs))
        return len(self._left_imgs)

    def get_left_intrinsics(self):
        return self._left_side_k

    def get_right_intrinsics(self):
        return self._right_side_k


def draw_obstacle_on_images(img, detections):
    for d in detections:
        tlx, tly, brx, bry = map(int, d['box'])
        if d['cls'] in [1, 3] and d['score'] > 0.5:
            cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 0, 255), 3)
    return img


if __name__ == '__main__':
    opts = Opt().init()
    if opts.is_plusai is False:
        dataset = RobotCarDataset(opts)
    else:
        dataset = PlusAIDataSet(opts)

    print len(dataset)
    cv2.namedWindow("obstacles")
    for item in dataset:
        left_image = item['left_image']
        # right_image = item['right_image']

        draw_obstacle_on_images(left_image, item['left_detection'])
        # draw_obstacle_on_images(right_image, item['right_detection'])

        # conv_image = np.hstack((left_image, right_image))
        conv_image = left_image
        cv2.imshow("obstacles", conv_image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit(0)
        elif key == ord('c'):
            continue
    cv2.destroyAllWindows()
