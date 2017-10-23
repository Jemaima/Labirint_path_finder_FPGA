import cv2
import numpy as np
import matplotlib.pyplot as plt
import path_finder as pf

# cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
fig = plt.gcf()
fig.show()
fig.canvas.draw()

memoryCapacity = 64
side_pad = 150
batch_step = 4
scan_step = 2
fill_step = 2

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    gray = pf.median_binary_filter(gray / 255)
    initialIm = np.copy(gray)
    img = np.copy(gray)
    primitiveSize = 0
    filters = []
    if np.sum(img[:, 0]) == 0 and np.sum(img[:, -1]) == 0 and np.sum(img[0, 0:side_pad])  == 0 and np.sum(img[0, -side_pad:]) == 0:
        for i in range(0, img.shape[0] - memoryCapacity,
                       batch_step):  # цикл по блокам из-за ограничения памяти
            # формируем список примитивов
            if i == 0:
                pathWidth = np.count_nonzero(img[0])
                primitiveSize = pathWidth * 2 + 1
                primitive = np.zeros((primitiveSize, primitiveSize))
                inv_primitive = np.zeros((primitiveSize, primitiveSize))
                primitive[pathWidth, -1] = 1
                inv_primitive[0] = [1] * primitiveSize
                inv_primitive[-1] = [1] * primitiveSize
                inv_primitive[:, 0] = [1] * primitiveSize
                filters = np.zeros(
                    [8, primitiveSize, primitiveSize])  # матрица, содержащая 4 одинаковых примитива разной ориентации
                filters[0] = primitive
                filters[4] = inv_primitive
                for f in range(1, 4):
                    filters[f] = pf.rotate_square_matrix(filters[f - 1])
                    filters[4 + f] = pf.rotate_square_matrix(filters[4 + f - 1])

            for y in range(0, memoryCapacity - primitiveSize, scan_step):  # цикл по строкам в блоке
                for x in range(side_pad, img.shape[1] - primitiveSize - side_pad,
                               scan_step):  # цикл по пикселям в строке
                    # area covered by filter
                    local_area = img[i + y:i + y + primitiveSize, x: x + primitiveSize]
                    for ind in range(4):
                        if ((np.sum(local_area * filters[ind]) == 1) and (np.sum(local_area * filters[ind + 4]) == 0)):
                            img = pf.painting_sleev(img, pathWidth, filters, i + y, x, ind)
                            break
                        else:
                            img[i + y: i + y + primitiveSize, x: x + primitiveSize] = img[i + y:i + y + primitiveSize,
                                                                                      x:x + primitiveSize]
        plt.imshow((img + initialIm) / 2)
        fig.canvas.draw()
    else:
        plt.imshow(initialIm)
        fig.canvas.draw()
cap.release()
cv2.destroyAllWindows()
