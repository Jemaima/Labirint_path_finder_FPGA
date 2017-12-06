import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage.morphology import binary_erosion

plt.ion()
dirs = [0, 1, 2, 3]

# 0 - down
# 1 - left
# 2 - up
# 3 - right

def img_preprocessing(img):
    img = intoGrayScale(img, False)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('initial image')
    # initialIm = np.copy(img)
    #
    # start_time = time.time()

    img = binary(img, 0.98)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('binary image')
    # img = binary_erosion(img).astype(img.dtype)
    # img = median_binary_filter(img, (3,3))
    # preprocessing_time = time.time()-start_time
    # print('binarization and filtration takes: ', str(preprocessing_time), 's')

    # plt.subplot(2, 1, 2)
    # plt.imshow(img)
    # plt.title('After median filtration image')
    plt.show()

    return img


def median_binary_filter(img, size=(3, 3)):
    img_f = np.zeros([img.shape[0],img.shape[1]], np.int32)
    for i in range(0, img.shape[0] - size[0], 1):
        for j in range(0, img.shape[1] - size[1], 1):
            if np.sum(img[i:i + size[0], j : j + size[1]]) > 0.8 * size[0] * size[1]:
                img_f[i:i + size[0], j : j + size[1]] = np.ones(size, np.int32)
            else:
                img_f[i:i + size[0], j : j + size[1]] = np.zeros(size, np.int32)
    return img_f


def intoGrayScale(im, add_noise=False):
    """ Формируем черно-белое изображение """
    grayscaling = lambda x: np.sqrt(np.sum(np.square(x[:-1])))
    greyIm = np.array(
        [np.array([grayscaling(a) for a in row]) for row in img])
    if add_noise:
        greyIm = greyIm + (np.random.normal(0, 0.1, im.shape[:-1]))
    return greyIm


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


def count_nods(im, corners, down, left, up, right):
    n = 0
    if corners:
        n += im[center - pathWidth, center - pathWidth] + im[center - pathWidth, center + pathWidth] + im[
            center + pathWidth, center - pathWidth] + im[center + pathWidth, center + pathWidth]
    if down:
        n += im[center + pathWidth, center]
    if left:
        n += im[center, center - pathWidth]
    if up:
        n += im[center - pathWidth, center]
    if right:
        n += im[center, center + pathWidth]
    return n


def check_possible_direction(local_area, cur_dir, possible_dirs=[1, 1, 1, 1]):
    if count_nods(local_area,True,False,False,False,False) == 0:
        try:
            if any(define_next_point(local_area, dirs[cur_dir - 1]) != [center, center]) and possible_dirs[
                dirs[cur_dir - 1]] != 0:
                return dirs[cur_dir - 1], define_next_point(local_area, dirs[cur_dir - 1])
            elif any(define_next_point(local_area, dirs[cur_dir]) != [center, center]) and possible_dirs[
                dirs[cur_dir]] != 0:
                return dirs[cur_dir], define_next_point(local_area, dirs[cur_dir])
            elif any(define_next_point(local_area, dirs[cur_dir + 1]) != [center, center]) and possible_dirs[
                dirs[cur_dir + 1]] != 0:
                return dirs[cur_dir + 1], define_next_point(local_area, dirs[cur_dir + 1])
            else:
                return -1,[center,center]
        except:
            if any(define_next_point(local_area, dirs[cur_dir - 1]) != [center, center]) and possible_dirs[
                dirs[cur_dir - 1]] != 0:
                return dirs[cur_dir - 1], define_next_point(local_area, dirs[cur_dir - 1])
            elif any(define_next_point(local_area, dirs[cur_dir]) != [center, center]) and possible_dirs[
                dirs[cur_dir]] != 0:
                return dirs[cur_dir], define_next_point(local_area, dirs[cur_dir])
            elif any(define_next_point(local_area, dirs[cur_dir - 3]) != [center, center]) and possible_dirs[
                dirs[cur_dir - 3]] != 0:
                return dirs[cur_dir - 3], define_next_point(local_area, dirs[cur_dir - 3])
            else:
                return -1,[center,center]

    else:
        return cur_dir, define_next_point(local_area, dirs[cur_dir])


def define_next_point(local_area, dir):
    width = 0
    pos = [center, center]

    if dir == 0:  # down
        for i in range(center - int(pathWidth), center + int(pathWidth)):
            if local_area[center + int(pathWidth), i] == 1:
                width += 1
                pos = [center + int(pathWidth / step_coef) + 1, i - int(width / 2) + 1]
    elif dir == 1:  # left
        for i in range(center - int(pathWidth), center + int(pathWidth)):
            if local_area[i, center - int(pathWidth)] == 1:
                width += 1
                pos = [i - int(width / 2), center - int(pathWidth / step_coef)]
    elif dir == 2:  # up
        for i in range(center - int(pathWidth), center + int(pathWidth)):
            if local_area[center - int(pathWidth), i] == 1:
                width += 1
                pos = [center - int(pathWidth / step_coef), i - int(width / 2)]
    elif dir == 3:
        for i in range(center - int(pathWidth), center + int(pathWidth)):
            if local_area[i, center + int(pathWidth)] == 1:
                width += 1
                pos = [i - int(width / 2), center + int(pathWidth / step_coef)]
    return np.array(pos)


def reset():
    global ENABLE, FOUND,path, pathWidth, pathWidth_bottom, n_shot, n_nodes
    FOUND = False
    ENABLE = False
    # NODE_ZONE = False
    pathWidth = 0
    pathWidth_bottom = 0
    path = np.ones([100, 7], dtype=np.int16)
    n_nodes = 0
    n_shot = 0


if __name__ == '__main__':
    ENABLE = False  # True if all labyrinth parameters defined and img is static
    FOUND = False  # True when in last path point
    # NODE_ZONE = False  # to prevent double nodes
    MEMORY_CAPACITY = 31
    center = int((MEMORY_CAPACITY - 1) / 2)
    # cur_dir = 0
    step_coef = 2
    path = np.ones([100, 7], dtype=np.int16)
    # [0,1] - coordinated, [2] - move dir when node was append, [3:6] - possible dirs

    n_nodes = 0

    # img = plt.imread('labyrint.png')
    img = plt.imread('shots/shot5.png')
    img = img_preprocessing(img)

    start_time = time.time()
    pathWidth = 0
    pathWidth_bottom = 0
    n_shot = 0
    a=0
    b= [0,0]

    while n_shot <= 1000000:
        img_out = np.array(img).copy()
        # ====================================
        # Define labyrinth parameters
        # ====================================
        if not ENABLE:  # define labirint parameters
            # scan image with

            for y_block in range(0, img_out.shape[0] - MEMORY_CAPACITY):  # цикл по блокам из-за ограничения памяти
                for x_block in range(0, img_out.shape[1] - MEMORY_CAPACITY):
                    if y_block == 0 and img_out[0, x_block + center] == 1:
                        pathWidth += 1
                        path[0] = [center, x_block + center - (int)(pathWidth / 2), 0, 1, 1, 1, 1]

                    elif y_block == img_out.shape[0] - MEMORY_CAPACITY - 1 and img_out[
                                y_block + center, x_block + center] == 1:
                        pathWidth_bottom += 1

                    else:
                        pass
            n_nodes += 1
            path[1] = path[0]
            if np.abs(pathWidth - pathWidth_bottom) < 4 and pathWidth != 0 and (
                            pathWidth * 2 + 1) < MEMORY_CAPACITY:
                pathWidth = int((pathWidth + pathWidth_bottom) / 2)

                # f_node_corners = np.zeros((MEMORY_CAPACITY, MEMORY_CAPACITY))
                # f_node_corners[center - pathWidth, center - pathWidth] = 1
                # f_node_corners[center + pathWidth, center - pathWidth] = 1
                # f_node_corners[center - pathWidth, center + pathWidth] = 1
                # f_node_corners[center + pathWidth, center + pathWidth] = 1

                # f_node_h = np.zeros((MEMORY_CAPACITY, MEMORY_CAPACITY))
                # f_node_h[center, center - pathWidth] = 1
                # f_node_h[center, center + pathWidth] = 1
                #
                # f_node_v = np.zeros((MEMORY_CAPACITY, MEMORY_CAPACITY))
                # f_node_v[center - pathWidth, center] = 1
                # f_node_v[center + pathWidth, center] = 1
                #
                # f_node = f_node_h + f_node_v
                img_out[path[0, 0]: pathWidth,
                path[0, 1] - int(pathWidth / 2):path[0, 1] + int(pathWidth / 2)] = 2
                ENABLE = True
            else:
                pass

        # ====================================
        # If parameters defined, try to STEP
        # ====================================
        else:
            # localStart = path[0][:2];
            # localPathWidth = 0;
            # Scan over the image, paint lines between nodes and find last point

            for y_block in range(0, img.shape[0] - MEMORY_CAPACITY):  # цикл по блокам из-за ограничения памяти
                for x_block in range(0, img.shape[1] - MEMORY_CAPACITY):

                    # check if smth changed
                    if all([y_block+center, x_block+center] == path[0][:2]) and img_out[y_block+center, x_block+center] == 0:
                        reset()

                    # if in last path point
                    if all([y_block + center, x_block + center] == path[n_nodes][:2]):
                        FOUND = True
                        local_area = img_out[y_block:y_block + MEMORY_CAPACITY, x_block: x_block + MEMORY_CAPACITY]
                        # Check deadend
                        a,b = check_possible_direction(local_area, path[n_nodes][2],path[n_nodes][3:])
                        b += np.array([y_block, x_block])
                        if check_possible_direction(local_area, path[n_nodes][2],
                                                    path[n_nodes][3:])[0] == -1 or count_nods(local_area,False,True,True,True,True) == 1:
                            # if all(define_next_point(local_area, path[n_nodes][2]) == [0, 0]):
                            n_nodes -= 1
                            path[n_nodes + 1] = [1] * 7

                        # Check streight
                        elif count_nods(local_area, True, False, True, False, True) == 0 or (
                                    count_nods(local_area, True, True, False, True, False) == 0):
                            path[n_nodes][:2] = define_next_point(local_area, path[n_nodes][2]) + np.array(
                                [y_block, x_block])
                            path[n_nodes][3:] = [1, 1, 1, 1]

                        # continue after/before node
                        elif count_nods(local_area, True, False, False, False, False) != 0:
                            path[n_nodes][path[n_nodes][2] % 2] = \
                                (define_next_point(local_area, path[n_nodes][2]) + np.array(
                                    [y_block, x_block]))[
                                    path[n_nodes][2] % 2]
                            path[n_nodes][3:] = [1, 1, 1, 1]

                        # перекресток или поворот
                        else:
                            step_coef = 1
                            n_nodes += 1
                            path[n_nodes][:] = path[n_nodes - 1][:]
                            path[n_nodes][2], path[n_nodes][:2] = check_possible_direction(local_area,
                                                                                           path[n_nodes][2],
                                                                                           path[n_nodes][3:])
                            path[n_nodes][:2] += np.array([y_block, x_block])
                            path[n_nodes - 1][3 + path[n_nodes][2]] = 0
                            try:
                                path[n_nodes - 1][3 + path[n_nodes-1][2]+2] = 0
                            except:
                                path[n_nodes - 1][3 + path[n_nodes - 1][2] - 2] = 0
                            step_coef = 2

                    if FOUND:
                        break

                if FOUND:
                    FOUND = False
                    break

        # if smth goes wrong and agent returned to start
        if all(path[n_nodes] == path[0]) and n_nodes != 1:
            reset()

        # if maze done
        if path[n_nodes][0] >= img_out.shape[0] - 2*pathWidth:
            reset()

        plt.clf()

        img_out[path[n_nodes, 0] - int(pathWidth / 2): path[n_nodes, 0] + int(pathWidth / 2),
        path[n_nodes, 1] - int(pathWidth / 2):path[n_nodes, 1] + int(pathWidth / 2)] = 2
        n_shot += 1
        start_time = time.time()
        plt.imshow(img_out)
        plt.title('dir: {0}, coord: {1}'.format(a,b))
        plt.draw()
        plt.pause(0.001)

    print('finding path takes: ', str(time.time() - start_time), 's')
