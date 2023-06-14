import multiprocessing
import os
import cv2 as cv
import math


def process_image(image):
    i, target, name = image
    print(i, target, name)
    img = cv.imread('../data/original/%s/%s' % (target, name))

    height, width, _ = img.shape
    img = cv.resize(img, (math.floor(width * 512 / height), 512))

    dir = '../data/resized/%s' % target
    if not os.path.isdir(dir):
        os.mkdir(dir)
    cv.imwrite(os.path.join(dir, name), img)


if __name__ == '__main__':
    if not os.path.isdir('../data/resized'):
        os.mkdir('../data/resized')

    images = []
    i = 0
    for path, subdirs, files in os.walk('../data/original'):
        for name in files:
            images.append((i, os.path.basename(path), name))
            new_path = os.path.join('../data/resized', images[-1][1])
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            i += 1

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        result = pool.map(process_image, images)
        pool.close()
        pool.join()
