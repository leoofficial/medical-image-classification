import os

import cv2 as cv

if __name__ == '__main__':

    dataset = 'train'
    # dataset = 'test'
    label = 'NORMAL'
    # label = 'PNEUMONIA'

    for filename in os.listdir(os.path.join('data', 'XRay', dataset, label)):
        try:
            image = cv.imread(os.path.join('data', 'XRay', dataset, label, filename), 0)
            height, width = image.shape
            if height < width:
                left = ((width - height) // 2)
                image = image[:, left:left + height]
            else:
                top = ((height - width) // 2)
                image = image[top:top + width, :]
            image = cv.resize(image, (64, 64), interpolation=cv.INTER_AREA)
            cv.imwrite(os.path.join('data', 'XRay', dataset, label, filename), image)
        except:
            pass
