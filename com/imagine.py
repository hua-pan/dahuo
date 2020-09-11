# coding:utf-8

import os
import cv2
import base64
import numpy as np

'''
image = ImageHandle.base64_cv2(base64_image)
for segment in ImageHandle.segment_image(image):
    print(ImageHandle.resize(segment))
'''


class Imagine:

    @staticmethod
    def image_from_base64(base64_image):
        image_str = base64.b64decode(base64_image)
        image = cv2.imdecode(np.frombuffer(image_str, np.uint8), cv2.IMREAD_GRAYSCALE)
        return image

    @staticmethod
    def image_from_file(path, limit=None):
        images, targets = [], []
        for target in os.listdir(path):
            target_path = path + target
            for file in os.listdir(target_path)[:limit]:
                file_path = target_path + '/' + file
                image = cv2.imread(file_path, 0)
                image = Imagine.padding_resize(image)
                images.append(image)
                targets.append(int(target))
        return images, targets

    @staticmethod
    def image_size(image):
        return image.shape

    @staticmethod
    def padding_resize(image, expect_w=20, expect_h=25):
        h, w = image.shape
        diff_w, diff_h = expect_w - w, expect_h - h
        if diff_w > 0 and diff_h > 0:
            image = cv2.copyMakeBorder(image, 0, diff_h, 0, diff_w, cv2.BORDER_CONSTANT, value=0)
        elif diff_w < 0 < diff_h:
            image = image[:, :expect_w]
            image = cv2.copyMakeBorder(image, 0, diff_h, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif diff_w > 0 > diff_h:
            image = image[:expect_h]
            image = cv2.copyMakeBorder(image, 0, 0, 0, diff_w, cv2.BORDER_CONSTANT, value=0)
        else:
            image = image[:expect_h, :expect_w]
        return image

    @staticmethod
    def letterbox_resize(image, expect_w=16, expect_h=16):
        h, w = image.shape[0:2]
        scale = min(expect_h / h, expect_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top = (expect_h - new_h) // 2
        bottom = expect_h - new_h - top
        left = (expect_w - new_w) // 2
        right = expect_w - new_w - left
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

    @staticmethod
    def draw_rect(image, rect_point, color=(255, 0, 0), thickness=1):
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for min_point, max_point in rect_point:
            cv2.rectangle(color_image, min_point, max_point, color, thickness)
        return color_image

    @staticmethod
    def denoise(image, noise_num=2, circle=2):
        _, denoise_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
        rows, cols = denoise_image.shape
        for _ in range(circle):
            temp_image = np.copy(denoise_image)
            for w in range(cols):
                for h in range(rows):
                    count = 0
                    points = [(w - 1, h - 1), (w, h - 1), (w + 1, h - 1),
                              (w - 1, h), (w + 1, h),
                              (w - 1, h + 1), (w, h + 1), (w + 1, h + 1)]

                    try:
                        for point in points:
                            count += bool(temp_image[point[1], point[0]])
                    except:
                        pass
                    if temp_image[h, w] and count <= noise_num:
                        denoise_image[h, w] = 0

        return denoise_image

    @staticmethod
    def divide(denoise_image, count=4, step=22):
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(denoise_image, 8, cv2.CV_8U)
        contours = np.concatenate((stats[1:, :2], stats[1:, 2:4] + stats[1:, :2]), 0)
        images = []
        for i in range(count):
            matrix = contours[np.where((contours[:, 0] > i * step) & (contours[:, 0] <= i * step + step))]
            min_point = (matrix[:, 0].min(), matrix[:, 1].min())
            max_point = (matrix[:, 0].max(), matrix[:, 1].max())
            images.append(denoise_image[min_point[1]:max_point[1] + 1, min_point[0]:max_point[0] + 1])
        return images
