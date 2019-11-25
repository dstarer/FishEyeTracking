import os
import sys

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from dataset import RobotCarDataset, draw_obstacle_on_images, PlusAIDataSet
from opts import Opt


def find_grounding_point(edge_image, box):
    """
    Given a bounding box (tlx, tly, brx, bry) and one canny image,
    find the concat point
    :param edge_image:
    :param box:
    :return:
    """
    tlx, tly, brx, bry = box
    min_row = bry - (bry - tly) / 10
    row = bry
    edges = []
    while row >= min_row:
        for col in range(tlx, brx):
            if edge_image[row, col] > 0:
                edges.append((col, row))
        row -= 1

    if len(edges) <= 0:
        return None

    edges = np.array(edges)
    v = np.average(edges, 0)
    return v[0], v[1]


def compute_dist(pixel, k_inv, theta=0, h=1.36):
    pt = np.array([[pixel[0]], [pixel[1]], [1]], dtype=np.float32)
    vec = np.array([[0, np.cos(theta), np.sin(theta)]], dtype=np.float32)
    z = np.matmul(np.matmul(vec, k_inv), pt)
    X = h / z * np.matmul(k_inv, pt)
    return X


def find_all_grounding_point(edge_image, detections):
    pts = []
    for d in detections:
        tlx, tly, brx, bry = map(int, d['box'])
        if d['cls'] in [1, 3] and d['score'] > 0.5:
            pt = find_grounding_point(edge_image, (tlx, tly, brx, bry))
            pts.append(pt)


def draw_all_grounding_point(edge_image, color_image, detections, K_inv, theta, height):
    for d in detections:
        tlx, tly, brx, bry = map(int, d['box'])
        if d['cls'] in [1, 3] and d['score'] > 0.5:
            pt = find_grounding_point(edge_image, (tlx, tly, brx, bry))
            cv2.rectangle(color_image, (tlx, tly), (brx, bry), (0, 0, 255), 3)
            if pt is not None:
                X = compute_dist(pt, K_inv, theta, height)
                print X
                text = "x: %.1f, y:%.1f, z:%.1f" % (X[0], X[1], X[2])
                cv2.putText(color_image, text, (int(tlx), int(tly / 2. + bry / 2.)),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.circle(color_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), 2)


def compute_edge_image(image, thresh_low=100, thresh_high=200):
    img = cv2.GaussianBlur(image, (3, 3), 0)
    canny = cv2.Canny(img, thresh_low, thresh_high)
    return canny


if __name__ == '__main__':
    opts = Opt().init()
    if not opts.is_plusai:
        dataset = RobotCarDataset(opts)
    else:
        dataset = PlusAIDataSet(opts)
    K_left = dataset.get_left_intrinsics()
    K_right = dataset.get_right_intrinsics()
    K_left_inv = np.linalg.inv(K_left)
    K_right_inv = np.linalg.inv(K_right)

    # theta_left = 50.0 / 180 * np.pi
    # theta_right = 35.0 / 180 * np.pi

    theta_left = 10.0 / 180 * np.pi
    theta_right = 10.0 / 180 * np.pi
    # height = 3.27
    height = 1.36

    print compute_dist((300, 400), K_left_inv, theta_left, height)

    cv2.namedWindow("obstacles")
    if opts.output is not None:
        if not os.path.exists(opts.output):
            os.mkdir(opts.output)

    for ind, item in enumerate(dataset):
        left_image = item['left_image']
        right_image = item['right_image']

        # draw_obstacle_on_images(left_image, item['left_detection'])
        # draw_obstacle_on_images(right_image, item['right_detection'])
        left_canny = compute_edge_image(left_image)
        right_canny = compute_edge_image(right_image)

        # draw_obstacle_on_images(left_canny, item['left_detection'])
        # draw_obstacle_on_images(right_canny, item['right_detection'])
        draw_all_grounding_point(left_canny, left_image, item['left_detection'], K_left_inv, theta_left, height)
        draw_all_grounding_point(right_canny, right_image, item['right_detection'], K_right_inv, theta_right, height)

        conv_image = np.hstack((left_image, right_image))
        # conv_image = left_image
        # conv_image = right_image
        if opts.output is not None:
            cv2.imwrite(os.path.join(opts.output, "%06d.jpg" % ind), conv_image)
        # cv2.imshow("obstacles", conv_image)

        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     exit(0)
        # elif key == ord('c'):
        #     continue

    cv2.destroyAllWindows()
