#!/usr/bin

from cv_bridge import CvBridge
import rosbag
import os
import cv2
import argparse
import numpy as np
import glob


class CameraModel:
    def __init__(self, imgs_path, suffix='.bmp'):

        self.DIM, self.K, self.D = self.get_K_and_D((6, 8), 82.75, imgs_path, suffix)

    def get_K_and_D(self, checkerboard, chessboardSize, imgsPath, format='.jpg'):

        CHECKERBOARD = checkerboard
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * chessboardSize
        _img_shape = None
        objpoints = []
        imgpoints = []
        images = glob.glob(imgsPath + '/*' + format)
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape is None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret is True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        DIM = _img_shape[::-1]
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
        return DIM, K, D

    def undistort(self, img):
        K = self.K
        D = self.D
        DIM = self.DIM
        scale = 1.0
        imshow = False
        dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert dim1[0] / dim1[1] == DIM[0] / DIM[
            1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if dim1[0] != DIM[0]:
            img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
        Knew = K.copy()
        if scale:  # change fov
            Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if imshow:
            cv2.imshow("undistorted", undistorted_img)
        return undistorted_img


def extract_imgs(bag_file, target_path, topic_name, camera_model):
    bridge = CvBridge()
    print '##', bag_file

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    bag = rosbag.Bag(bag_file, 'r')
    print 'Extract image...'
    try:
        for topic, msg, bag_time in bag.read_messages(topics=[topic_name]):
            t = msg.header.stamp.to_sec()
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            undistorted = camera_model.undistort(cv_image)
            print "saved ", os.path.join(target_path, "%.3f.jpg" % t)
            cv2.imwrite(os.path.join(target_path, '%.3f.jpg' % t), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    finally:
        bag.close()


args = argparse.ArgumentParser()
args.add_argument('--bag', type=str, help="bags")
args.add_argument('--calib_dir', type=str, help="calibration models")
args.add_argument("--output", type=str, help="output directory")
args.add_argument("--topic", type=str, default="/side_left_camera/image_raw/compressed", help="topic of camera")

opts = args.parse_args()

camera_model = CameraModel(opts.calib_dir)

output_path = os.path.join(opts.output, opts.bag.split(".")[0])
print output_path

if not os.path.exists(output_path):
    os.mkdir(output_path)

output_path = os.path.join(output_path, opts.topic.split("/")[1])
print output_path
if not os.path.exists(output_path):
    os.mkdir(output_path)

extract_imgs(opts.bag, output_path, opts.topic, camera_model)
