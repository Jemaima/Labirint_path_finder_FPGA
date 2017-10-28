import numpy as np
import matplotlib.pyplot as plt
import time

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

    # img = median_binary_filter(img, (3,3))
    # preprocessing_time = time.time()-start_time
    # print('binarization and filtration takes: ', str(preprocessing_time), 's')

    # plt.subplot(2, 1, 2)
    # plt.imshow(img)
    # plt.title('After median filtration image')
    plt.show()

    return img


def check_straight(local_area, p_w):
    if (sum(local_area[center - p_w, center - p_w:center + p_w]) == 0 and sum(
            local_area[center + p_w, center - p_w:center + p_w]) == 0) or (
                    sum(
                        local_area[center - p_w:center + p_w, center - p_w]) == 0 and sum(
                local_area[center - p_w:center + p_w, center + p_w]) == 0):
        return True
    else:
        return False


def check_possible_direction(local_area, cur_dir, p_w, f_node2, possible_dirs=[1, 1, 1, 1]):
    if sum(sum(local_area * f_node2)) == 0 or local_area[center, center] == 0:
        try:
            if all(define_next_point(local_area, dirs[cur_dir - 1], p_w) != [0, 0]) and possible_dirs[
                dirs[cur_dir - 1]] != 0:
                return dirs[cur_dir - 1], define_next_point(local_area, dirs[cur_dir - 1], p_w)
            elif all(define_next_point(local_area, dirs[cur_dir], p_w) != [0, 0]) and possible_dirs[dirs[cur_dir]] != 0:
                return dirs[cur_dir], define_next_point(local_area, dirs[cur_dir], p_w)
            elif all(define_next_point(local_area, dirs[cur_dir + 1], p_w) != [0, 0]) and possible_dirs[
                dirs[cur_dir + 1]] != 0:
                return dirs[cur_dir + 1], define_next_point(local_area, dirs[cur_dir + 1], p_w)
            else:
                return -1
        except:
            if all(define_next_point(local_area, dirs[cur_dir - 1], p_w) != [0, 0]) and possible_dirs[
                dirs[cur_dir - 1]] != 0:
                return dirs[cur_dir - 1], define_next_point(local_area, dirs[cur_dir - 1], p_w)
            elif all(define_next_point(local_area, dirs[cur_dir], p_w) != [0, 0]) and possible_dirs[dirs[cur_dir]] != 0:
                return dirs[cur_dir], define_next_point(local_area, dirs[cur_dir], p_w)
            elif all(define_next_point(local_area, dirs[cur_dir - 3], p_w) != [0, 0]) and possible_dirs[
                dirs[cur_dir - 3]] != 0:
                return dirs[cur_dir - 3], define_next_point(local_area, dirs[cur_dir - 3], p_w)
            else:
                return -1

    else:
        return dirs[cur_dir], define_next_point(local_area, dirs[cur_dir - 1], p_w)


def define_next_point(local_area, dir, p_w):
    width = 0
    pos = [0, 0]
    if dir == 0:  # down
        for i in range(center - int(p_w), center + int(p_w)):
            if local_area[center + int(p_w), i] == 1:
                width += 1
                pos = [center + int(p_w / step_coef) + 1, i - int(width / 2) + 1]
    elif dir == 1:  # left
        for i in range(center - int(p_w), center + int(p_w)):
            if local_area[i, center - int(p_w)] == 1:
                width += 1
                pos = [i - int(width / 2), center - int(p_w / step_coef)]
    elif dir == 2:  # up
        for i in range(center - int(p_w), center + int(p_w)):
            if local_area[center - int(p_w), i] == 1:
                width += 1
                pos = [center - int(p_w / step_coef), i - int(width / 2)]
    elif dir == 3:
        for i in range(center - int(p_w), center + int(p_w)):
            if local_area[i, center + int(p_w)] == 1:
                width += 1
                pos = [i - int(width / 2), center + int(p_w / step_coef)]
    if width == 0:
        return np.array([0, 0])
    else:
        return np.array(pos)


def reset():
    global ENABLE, path, pathWidth, pathWidth_bottom, n_shot, n_nodes
    ENABLE = False
    pathWidth = 0
    pathWidth_bottom = 0
    path = np.ones([50, 7], dtype=np.int16)
    n_nodes = 0
    n_shot = 0


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




ENABLE = False  # true if all labyrinth parameters defined and img is static
FOUND = False
MEMORY_CAPACITY = 31
center = int((MEMORY_CAPACITY - 1) / 2)
# cur_dir = 0
step_coef = 1
path = np.ones([50, 7], dtype=np.int16)
# [0,1] - coordinated, [2] - move dir when node was append, [3:6] - possible dirs

n_nodes = 0

if __name__ == '__main__':

    # img = plt.imread('labyrint.png')
    img = plt.imread('shots/shot5.png')
    img = img_preprocessing(img)

    start_time = time.time()
    pathWidth = 0
    pathWidth_bottom = 0
    n_shot = 0

    while n_shot <= 100000:
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
            if check_straight(
                    img_out[path[n_nodes][0] - center:path[n_nodes][0] + center - 1,
                    path[n_nodes][1] - center:path[n_nodes][1] + center - 1], pathWidth):

                n_nodes += 1
                path[1] = path[0]
                if np.abs(pathWidth - pathWidth_bottom) < 4 and pathWidth != 0 and (
                                pathWidth * 2 + 1) < MEMORY_CAPACITY:
                    pathWidth = int((pathWidth + pathWidth_bottom) / 2)

                    f_node = np.zeros((MEMORY_CAPACITY, MEMORY_CAPACITY))
                    f_node[center - pathWidth, center] = 1
                    f_node[center + pathWidth, center] = 1
                    f_node[center, center - pathWidth] = 1
                    f_node[center, center + pathWidth] = 1

                    f_node2 = np.zeros((MEMORY_CAPACITY, MEMORY_CAPACITY))
                    f_node2[center - pathWidth, center - pathWidth] = 1
                    f_node2[center + pathWidth, center - pathWidth] = 1
                    f_node2[center - pathWidth, center + pathWidth] = 1
                    f_node2[center + pathWidth, center + pathWidth] = 1

                    img_out[path[0, 0]: pathWidth,
                    path[0, 1] - int(pathWidth / 2):path[0, 1] + int(pathWidth / 2)] = 2
                    ENABLE = True
                else:
                    pass

        # ====================================
        # If parameters defined, try to STEP
        # ====================================
        else:
            localStart = path[0][:2];
            localPathWidth = 0;
            # Scan over the image, paint lines between nodes and find last point
            for y_block in range(0, img.shape[0] - MEMORY_CAPACITY):  # цикл по блокам из-за ограничения памяти
                for x_block in range(0, img.shape[1] - MEMORY_CAPACITY):
                    if y_block == 0 and img_out[0, x_block + center] == 1:
                        localPathWidth += 1
                        localStart = [center, x_block + center - (int)(pathWidth / 2)]

                    # if in last path point
                    if all([y_block + center, x_block + center] == path[n_nodes][:2]):
                        FOUND = True
                        local_area = img_out[y_block:y_block + MEMORY_CAPACITY, x_block: x_block + MEMORY_CAPACITY]

                        # Check deadend
                        if check_possible_direction(local_area, path[n_nodes][2], pathWidth, f_node2, path[n_nodes][3:]) == -1 or \
                                        local_area[center, center] == 0:
                            path[n_nodes] = path[n_nodes - 1]
                            n_nodes -= 1
                            path[n_nodes + 2] = [1] * 7
                        elif check_straight(local_area, pathWidth):
                            path[n_nodes][:2] = define_next_point(local_area, path[n_nodes][2], pathWidth) + np.array(
                                [y_block, x_block])
                            path[n_nodes][3:] = [1, 1, 1, 1]
                        elif sum(sum(local_area * f_node2)) != 0:
                            path[n_nodes][path[n_nodes][2] % 2] = \
                                (define_next_point(local_area, path[n_nodes][2], pathWidth) + np.array(
                                    [y_block, x_block]))[
                                    path[n_nodes][2] % 2]
                            path[n_nodes][3:] = [1, 1, 1, 1]

                        else:

                            n_nodes += 1
                            path[n_nodes][:4] = path[n_nodes - 1][:4]
                            path[n_nodes][2], path[n_nodes][:2] = check_possible_direction(local_area, path[n_nodes][2],
                                                                                           pathWidth, f_node2, path[n_nodes][3:])
                            path[n_nodes][:2] += np.array([y_block, x_block])
                            path[n_nodes - 1][3 + path[n_nodes][2]] = 0
                            # path[n_nodes][3:] = [1, 1, 1, 1]
                    if FOUND:
                        break
                if sum(localStart - path[0][:2]) > 3:
                    reset()

                if FOUND:
                    FOUND = False
                    break

        if (img.shape[0] - path[n_nodes, 0]) < pathWidth:
            reset()

        plt.clf()

        img_out[path[n_nodes, 0] - int(pathWidth / 2): path[n_nodes, 0] + int(pathWidth / 2),
        path[n_nodes, 1] - int(pathWidth / 2):path[n_nodes, 1] + int(pathWidth / 2)] = 2
        n_shot += 1
        start_time = time.time()
        plt.imshow(img_out)
        plt.title(str(n_nodes))
        plt.draw()
        plt.pause(0.001)

    print('finding path takes: ', str(time.time() - start_time), 's')
