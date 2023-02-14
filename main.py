import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy

from nets.nn import FaceDetector

warnings.filterwarnings("ignore")


def main():
    parser = ArgumentParser()
    parser.add_argument('model', help='model file path')
    parser.add_argument('filepath', help='image file path')

    args = parser.parse_args()
    detector = FaceDetector(args.model)

    image = cv2.imread(args.filepath)
    boxes, points = detector.detect(image, score_thresh=0.5, input_size=(640, 640))
    for box in boxes:
        x1, y1, x2, y2, score = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if points is not None:
        for point in points:
            for kp in point:
                kp = kp.astype(numpy.int)
                cv2.circle(image, tuple(kp), 1, (0, 255, 0), 2)
    if not os.path.exists('./demo'):
        os.makedirs('./demo')
    cv2.imwrite(f'./demo/{os.path.basename(args.filepath)}', image)


if __name__ == '__main__':
    main()
