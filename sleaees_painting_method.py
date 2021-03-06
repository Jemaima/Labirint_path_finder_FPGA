import numpy as np
import matplotlib.pyplot as plt
import time


def intoGrayScale(im, add_noise=False):
    """ Формируем черно-белое изображение """
    grayscaling = lambda x: np.sqrt(np.sum(np.square(x[:-1])))
    greyIm = np.array(
        [np.array([grayscaling(a) for a in row]) for row in img])
    if add_noise:
        greyIm = greyIm + (np.random.normal(0, 0.1, im.shape[:-1]))
    return greyIm


def median_filter(img, size=(3, 3)):
    img_f = np.zeros([img.shape[0],img.shape[1]], np.int32)
    for i in range(0, img.shape[0] - size[0], 1):
        for j in range(0, img.shape[1] - size[1], 1):
            if np.sum(img[i:i + size[0], j : j + size[1]]) > 0.8 * size[0] * size[1]:
                img_f[i:i + size[0], j : j + size[1]] = np.ones(size, np.int32)
            else:
                img_f[i:i + size[0], j : j + size[1]] = np.zeros(size, np.int32)
    return img_f


def check_filteres(filters):
    n = 0
    plt.subplot(4, 2, 1)
    plt.imshow(filters[n])
    plt.subplot(4, 2, 2)
    plt.imshow(filters[4 + n])
    n += 1

    plt.subplot(4, 2, 3)
    plt.imshow(filters[n])
    plt.subplot(4, 2, 4)
    plt.imshow(filters[4 + n])
    n += 1

    plt.subplot(4, 2, 5)
    plt.imshow(filters[n])
    plt.subplot(4, 2, 6)
    plt.imshow(filters[4 + n])
    n += 1

    plt.subplot(4, 2, 7)
    plt.imshow(filters[n])
    plt.subplot(4, 2, 8)
    plt.imshow(filters[4 + n])


def binary(im, treshold=0.5):
    """
    Бинаризация по заданному порогу

    Parameters
    ----------
    im : array_like
        входное ч/б изображение.
    Returns
    -------
    binaryIm : ndarray
    """

    tresholdBinary = lambda x: 0 if x < treshold else 1
    binaryIm = np.array(
        [np.array([tresholdBinary(a) for a in row]) for row in im])
    return binaryIm


def rotate_square_matrix(mat):
    """
    Преобразование исходного примитива.
    Поворот двухмерной матрицы на 90 градусов
    """
    if (len(mat.shape) > 2 or (mat.shape[0] != mat.shape[1])):
        return mat
    else:
        mat2 = np.copy(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                mat2[i, j] = mat[mat.shape[1] - j - 1, i]
        return mat2


def painting_sleev(outIm, pathWidth, filters, y_loc, x_loc, ind):
    primitiveSize = filters[0].shape[0]
    if ((np.sum(outIm[y_loc: y_loc + primitiveSize, x_loc: x_loc + primitiveSize] * filters[ind]) == 1) and (
                np.sum(outIm[y_loc: y_loc + primitiveSize, x_loc: x_loc + primitiveSize] * filters[ind + 4]) == 0)):
        outIm[y_loc: y_loc + primitiveSize, x_loc: x_loc + primitiveSize] = np.zeros([primitiveSize, primitiveSize])
        if ind == 0:
            outIm = painting_sleev(outIm, pathWidth,  filters, y_loc, x_loc + pathWidth, ind)
        elif ind == 1:
            outIm = painting_sleev(outIm, pathWidth,  filters, y_loc + pathWidth, x_loc, ind)
        elif ind == 2:
            outIm = painting_sleev(outIm, pathWidth,  filters, y_loc, x_loc - pathWidth, ind)
        elif ind == 3:
            outIm = painting_sleev(outIm, pathWidth,  filters, y_loc - pathWidth, x_loc, ind)
        else:
            outIm = outIm

    return outIm


memoryCapacity = 64
side_pad = 150
batch_step = 8
scan_step = 2
fill_step = 2

if __name__ == '__main__':

    img = plt.imread('shots\shot4.png')
    img = intoGrayScale(img, False)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title('initial image')
    initialIm = np.copy(img)

    start_time = time.time()

    img = binary(img, 0.98)
    img = median_filter(img, (3,3))
    preprocessing_time = time.time()-start_time
    print('binarization and filtration takes: ', str(preprocessing_time), 's')

    plt.subplot(2, 1, 2)
    plt.imshow(img)
    plt.title('After median filtration image')
    plt.show()

    start_time = time.time()
    pathWidth = 0
    # outIm = np.copy(img)

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
                filters[f] = rotate_square_matrix(filters[f - 1])
                filters[4 + f] = rotate_square_matrix(filters[4 + f - 1])

        for y in range(0, memoryCapacity - primitiveSize, scan_step):  # цикл по строкам в блоке
            for x in range(side_pad, img.shape[1] - primitiveSize - side_pad, scan_step):  # цикл по пикселям в строке
                # area covered by filter
                local_area = img[i + y:i + y + primitiveSize, x: x + primitiveSize]
                for ind in range(4):
                    if ((np.sum(local_area * filters[ind]) == 1) and (np.sum(local_area * filters[ind + 4]) == 0)):
                        img = painting_sleev(img, pathWidth,  filters, i + y, x, ind)
                        break
                    else:
                        img[i + y: i + y + primitiveSize, x: x + primitiveSize] = img[i + y:i + y + primitiveSize,
                                                                                  x:x + primitiveSize]
    print('finding path takes: ', str(time.time() - start_time), 's')
    plt.subplot(1, 2, 2)
    plt.imshow((img + initialIm) / 2)
    plt.subplot(1, 2, 1)
    plt.imshow(initialIm)
    plt.show()
