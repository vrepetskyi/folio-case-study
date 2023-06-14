import cv2 as cv
import numpy as np
import os
import multiprocessing
import threading
import csv

csv_lock = threading.Lock()


def create_radial_shadow_mask(h, w, radius=None, center=None):
    # based on https://stackoverflow.com/a/44874588
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    mask = dist_from_center / dist_from_center[center[0] - radius, center[1]]
    return mask


def find_dominant_color(img, mask=None):
    a2D = img.reshape(-1, img.shape[-1])
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    frequencies = np.bincount(a1D, mask if mask is None else mask.flatten())
    return np.array(np.unravel_index(frequencies.argmax(), col_range))


input_folder = "../data/original"
output_folder = "../data/test"


def process_image(image):
    i, target_class, example_name = image
    print(i, target_class, example_name)
    img = cv.imread(os.path.join(input_folder, target_class, example_name))

    demo = False
    if demo:
        height, width, _ = img.shape
        img = cv.resize(img, (int(width * 512 / height), 512))

    radial_shadow_mask = create_radial_shadow_mask(*img.shape[:2])
    background_color = find_dominant_color(img, radial_shadow_mask)
    if demo:
        cv.imshow("Color weight mask", radial_shadow_mask)

    radial_spotlight_mask = None
    with np.errstate(divide="ignore"):
        radial_spotlight_mask = 1 / radial_shadow_mask
    foreground_color = find_dominant_color(img, radial_spotlight_mask)

    # TODO: try to decrease lower bound
    # try to use with lighthen non-tinted

    foreground_mask = cv.inRange(img, foreground_color * 0.6, foreground_color * 1.4)
    if demo:
        cv.imshow("Masked background", foreground_mask)

    background_mask = cv.inRange(img, background_color * 0.6, background_color * 1.4)
    if demo:
        cv.imshow("Masked foreground", background_mask)

    blur_amount = int(img.shape[0] * 0.0025)
    blured_mask = cv.blur(background_mask, (blur_amount, blur_amount))
    if demo:
        cv.imshow("Masked foreground + blur", blured_mask)

    _, thresh = cv.threshold(blured_mask, 127, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = sorted(contours, key=cv.contourArea)[-min(2, len(contours))]
    if demo:
        cv.imshow("Image with contour", img)

    to_save = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    to_save = cv.cvtColor(to_save, cv.COLOR_GRAY2BGR)
    contour_width = blur_amount
    cv.drawContours(to_save, contour, -1, (255, 0, 0), contour_width)

    # TODO: extract features from the contour, write them to the csv
    # write color to the csv
    # try to extract texture

    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = h / w
    cv.rectangle(to_save, (x, y), (x + w, y + h), (0, 255, 0), contour_width)

    contourPerimeter = cv.arcLength(contour, True)
    contourArea = cv.contourArea(contour)
    ellipse = cv.fitEllipse(contour)
    a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
    ellipsePerimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    ellipseArea = np.pi * a * b
    cv.ellipse(to_save, ellipse, (0, 0, 255), contour_width)

    smoothness = ellipsePerimeter / contourPerimeter

    boxArea = w * h
    shape_density = contourArea / boxArea

    roundness = a / b

    dir = os.path.join(output_folder, target_class)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    cv.imwrite(os.path.join(dir, example_name), to_save)

    with csv_lock:
        with open(
            os.path.join(output_folder, "dataset.csv"), "a", encoding="UTF8", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    example_name,
                    aspect_ratio,
                    smoothness,
                    shape_density,
                    roundness,
                    foreground_color[-1],
                    foreground_color[-2],
                    foreground_color[-3],
                    target_class,
                )
            )

    if demo:
        cv.waitKey()


if __name__ == "__main__":
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    header = (
        "example",
        "aspect_ratio",
        "smoothness",
        "shape_density",
        "roundness",
        "primary_red",
        "primary_green",
        "primary_blue",
        "target",
    )

    with open(
        os.path.join(output_folder, "dataset.csv"), "w", encoding="UTF8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)

    tested_example = None  # "data/test/rose/20150523_153254.jpg"
    if tested_example is not None:
        process_image((0, *os.path.normpath(tested_example).split(os.path.sep)[-2:]))
    else:
        images = []
        i = 0
        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                # if i < 0 or i > 10:
                #     continue
                images.append((i, os.path.basename(path), name))
                new_path = os.path.join(output_folder, images[-1][1])
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                i += 1

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(process_image, images)
            pool.close()
            pool.join()
