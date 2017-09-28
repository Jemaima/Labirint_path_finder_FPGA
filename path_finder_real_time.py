import cv2
import numpy as np
import matplotlib.pyplot as plt
from path_finder import painting_sleev, rotate_square_matrix

imSize = [256, 256]
shotSize = [256, 256]
memoryCapacity = 64

# cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
while True:
    ret, intial_img = cap.read()
    _, img = cv2.threshold(cv2.cvtColor(intial_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    # for i in range(0, shotSize[0] - memoryCapacity, int(memoryCapacity / 8)):
    #     if i == 0:
    #         pathWidth = np.count_nonzero(img[0])
    #         primitiveSize = pathWidth * 2 + 1
    #         primitive = np.zeros((primitiveSize, primitiveSize))
    #         inv_primitive = np.zeros((primitiveSize, primitiveSize))
    #         primitive[pathWidth, -1] = 1
    #         inv_primitive[0] = [1] * primitiveSize
    #         inv_primitive[-1] = [1] * primitiveSize
    #         inv_primitive[:, 0] = [1] * primitiveSize
    #         filters = np.zeros(
    #             [8, primitiveSize, primitiveSize])
    #         filters[0] = primitive
    #         filters[4] = inv_primitive
    #         for f in range(1, 4):
    #             filters[f] = rotate_square_matrix(filters[f - 1])
    #             filters[4 + f] = rotate_square_matrix(filters[4 + f - 1])
    #
    #     for y in range(0, memoryCapacity - primitiveSize, 1):
    #         for x in range(0, imSize[1] - primitiveSize, 1):
    #             # area covered by filter
    #             local_area = img[i + y:i + y + primitiveSize, x: x + primitiveSize]
    #             for ind in range(4):
    #                 if ((np.sum(local_area * filters[ind]) == 1) and (np.sum(local_area * filters[ind + 4]) == 0)):
    #                     img = painting_sleev(img, i + y, x, ind)
    #                     break
    #                 else:
    #                     img[i + y: i + y + primitiveSize, x: x + primitiveSize] = img[i + y:i + y + primitiveSize,
    #                                                                               x:x + primitiveSize]

    cv2.imshow('img', cv2.flip(img, 1))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()