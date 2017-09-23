import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


def intoGrayScale(im):
    grayscaling = lambda x: np.sqrt(np.sum(np.square(x[:-1])))
    greyIm = np.array(
        [np.array([grayscaling(a) for a in row]) for row in img]) + (np.random.normal(0, 0.5, im.shape[:-1]))
    return greyIm


def binary(im, treshold=0.5):
    tresholdBinary = lambda x: 0 if x < treshold else 1
    binaryIm = np.array(
        [np.array([tresholdBinary(a) for a in row]) for row in im])
    return binaryIm


imSize = [256, 512]
shotSize = [256, 256]
memoryCapacity = 64

if __name__ == '__main__':

    img = plt.imread('labirint.png')
    gray = intoGrayScale(img)
    plt.figure()
    plt.imshow(gray)
    plt.title('initial image')

    binImg = binary(gray)
    plt.figure()
    plt.imshow(binImg)
    plt.title('binary image')
    plt.show()

    img = binImg
    outIm = np.copy(img)
    pathWidth = 0
    primitiv = np.zeros((5, 5))
    for i in range(shotSize[0] - memoryCapacity):
        partIm = img[i:i + memoryCapacity, :]
        partIm = ndimage.grey_closing(ndimage.grey_erosion(partIm, size=(3, 3)), size=(15, 15))
        plt.imshow(partIm)

        if i == 0:
            pathWidth = np.count_nonzero(partIm[0])
            primitivSize = pathWidth * 2 + 1
            primitiv = np.zeros((primitivSize, primitivSize))
            primitiv[0, pathWidth], primitiv[-1, pathWidth], primitiv[pathWidth, 0], primitiv[
                pathWidth, -1] = 1, 1, 1, 1

        for y in range(memoryCapacity - primitivSize):
            for x in range(imSize[1] - primitivSize):
                if np.sum(partIm[y + primitivSize, x + primitivSize] * primitiv) <= 1:
                    outIm[i + y : i+y+ primitivSize, x: x + primitivSize] = ndimage.grey_erosion(partIm[y:y + primitivSize, x :x+ primitivSize], size=(primitivSize, primitivSize))
                else:
                    outIm[i + y: i + y + primitivSize, x: x + primitivSize] = partIm[y:y + primitivSize, x :x+ primitivSize]

    plt.imshow(outIm)
    plt.show()
