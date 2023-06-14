import multiprocessing
import os
import cv2 as cv
import numpy as np


def lighten_non_tinted(img):
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            p = img[r, c]
            channelMax = np.max(p)
            channelMin = np.min(p)
            channelSum = np.sum(p)
            if channelMax - channelMin < 10 and channelSum > 275:
                img[r, c] = (255, 255, 255)


def process_image(image):
    i, target, name = image
    print(i, target, name)
    img = cv.imread('../data/resized/%s/%s' % (target, name))

    lighten_non_tinted(img)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img, 170, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = sorted(contours, key=cv.contourArea)[-min(2, len(contours))]

    cv.drawContours(img, contour, -1, (0, 255, 0), 3)

    dir = '../data/base-contour/%s' % target
    if not os.path.isdir(dir):
        os.mkdir(dir)
    cv.imwrite(os.path.join(dir, name), img)


if __name__ == '__main__':
    if not os.path.isdir('../data/base-contour'):
        os.mkdir('../data/base-contour')

    images = []
    i = 0
    for path, subdirs, files in os.walk('../data/original'):
        for name in files:
            images.append((i, os.path.basename(path), name))
            new_path = os.path.join('../data/base-contour', images[-1][1])
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            i += 1

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        result = pool.map(process_image, images)
        pool.close()
        pool.join()
