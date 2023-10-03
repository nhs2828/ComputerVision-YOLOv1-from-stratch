import cv2
import numpy as np
import collections
import skimage as ski
import random
import json
import os
import shutil


Label = collections.namedtuple('Label', ['classe', 'x', 'y', 'w', 'h']) # x-center, y-center, width, height


class OverlapException(Exception):
    pass

MIN_LEN_SQUARE = 4
MIN_RADIUS = 5


def createSquare(image, x, y, min_len_side=MIN_LEN_SQUARE):
    """
        @image          : image on which we want to draw a square
        @x, y0        : top left coordonate of the square
        @min_len_side   : min value for side
    """
    max_height, max_longer, depth = image.shape
    max_len_side = min(max_height-y, max_longer-x)

    if max_len_side > min_len_side:
        len_side = np.random.randint(min_len_side, max_len_side) # get a random side value
    else:
        len_side = min_len_side

    x1 = x+len_side
    y1 = y+len_side
    # Check overlap
    if np.min(image[x:x1, y:y1]) < 255: # if not white
        raise OverlapException()
    # if no overlap
    if depth == 1: # gray
        image[x:x1,y:y1]=0
    x_center = (x+x1)/2.
    y_center = (y+y1)/2.
    return Label('square', x_center, y_center, len_side, len_side)


def createCircle(image, x, y, min_radius=MIN_RADIUS):
    """
        @image          : image on which we want to draw a circle
        @x, y0        : center coord of the circle
        @min_radius     : min value for radius
    """
    max_height, max_longer, depth = image.shape
    max_radius = min(x, y, max_height-y, max_longer-x)//2
    while True:
        if max_radius > min_radius:
            radius = np.random.randint(min_radius, max_radius)
        else:
            radius = min_radius
        rr, cc = ski.draw.disk((x, y), radius=radius)
        # if the values are negatives, we would draw the circle to the other side
        if radius == min_radius and ((rr==max_longer).sum() != 0 or (cc==max_height).sum()!=0 or (rr<0).sum() != 0 or (cc<0).sum()!=0):
            raise OverlapException()
        if (rr==max_longer).sum() == 0  and (cc==max_height).sum()==0 and (rr<0).sum() == 0  and (cc<0).sum()==0:
            break
        print(f"Stuck here circle: rad = {radius}, min_rad = {min_radius}. NB neg {(rr<0).sum()}, {(cc<0).sum()}")
    # Check overlap
    if np.min(image[rr, cc]) < 255: # if not white
    # if (image[rr, cc] == 255).sum() != len(image[rr, cc]): # if not white
        raise OverlapException()
    # if no overlap
    if depth == 1: # gray
        image[rr, cc] = 0
    return Label('circle', x, y, 2*radius, 2*radius)


GENERATOR = [createSquare, createCircle]

def createImage(width, height, depth, nb_object):
    img = 255*np.ones((width, height, depth)).astype(np.uint8)
    labels = []
    for _ in range(nb_object):
        x = np.random.randint(width-min(MIN_LEN_SQUARE,MIN_RADIUS))
        y = np.random.randint(height-min(MIN_LEN_SQUARE,MIN_RADIUS))

        gen = random.choice(GENERATOR)
        # gen = createSquare
        attemp = 20
        while attemp > 0:
            try:
                label = gen(img, x, y)
            except OverlapException:
                attemp -= 1
            else:
                labels.append(label)
                break
    return img, labels


def genData(path, nb_Img, width, height, depth):
    print("Generation in process")
    Images = []
    Labels = []
    for i in range(nb_Img):
        print(f"img{i}")
        nb_object = np.random.randint(3,5)
        img, labels = createImage(width=width, height=height, depth=depth, nb_object=nb_object)
        Images.append(img)
        Labels.append(labels)
    print("Generation: done")
    if os.path.exists(path) and os.path.isdir(path):
        if os.listdir(path) == 0: # empty folder
            os.remove(path)
        else:
            shutil.rmtree(path)
    else:
        os.makedirs(path)
    print("Writing data in process")
    print("Writing images")
    for i in range(len(Images)):
        cv2.imwrite(path+f'img{i}.png', Images[i])
    path_labels = os.path.join(path, 'annotation.json')
    # path_labels = path+'annotation.json'
    print("Writing annotations")
    with open(path_labels, 'w') as f_out:
        f_out_labels = []
        for img_labels in Labels:
            list_labels_img = []
            for object_label in img_labels:
                list_labels_img.append(object_label._asdict())
            f_out_labels.append(dict(bboxes=list_labels_img))
        json.dump(f_out_labels, f_out)
    print("Writing data: done")


if __name__ == '__main__':
    # im, _ = createImage(400, 400, 1, 5)
    # print("created")
    # while True:
    #     cv2.imshow("test", im)
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    nbImg = 1000
    width = 400
    height = 400
    depth = 1
    path = 'data/'
    genData(path=path,\
            nb_Img=nbImg,\
                width=width,\
                    height=height,\
                        depth=depth)

    



