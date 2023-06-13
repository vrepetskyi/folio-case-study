import multiprocessing
import os
import cv2 as cv
import numpy as np


images = []
i = 0
for path, subdirs, files in os.walk('../data/original'):
    for name in files:
        images.append((i, os.path.basename(path), name))
        i += 1


def crop_image(image):
    i, target, name = image
    print(i, target, name)

    img = cv.imread('../data/original/%s/%s' % (target, name))

    height, width, _ = img.shape
    x = height if height > width else width
    y = height if height > width else width
    square = np.full((x, y, 3), 255, np.uint8)
    square[int((y-height)/2):int(y-(y-height)/2),
           int((x-width)/2):int(x-(x-width)/2)] = img

    cropped = cv.resize(square, (512, 512))

    dir = '../data/cropped/%s' % target
    if not os.path.isdir(dir):
        os.mkdir(dir)
    cv.imwrite(os.path.join(dir, name), cropped)


if __name__ == '__main__':
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        result = pool.map(crop_image, images)
        pool.close()
        pool.join()
