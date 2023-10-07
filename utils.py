import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import copy

IMG_SIZE = 448
NB_CELL = 7
CELL_SIZE = IMG_SIZE/NB_CELL
NB_BOXES = 2
DEPTH = 3
NB_CLASSES = 4
ALPHA_COORD = 5
ALPHA_NOOBJ = 0.5
CLASSES = {0: 'prohibitory', 1: 'danger', 2: 'mandatory', 3: 'other'} # well, there is a function below for this, better for notebook


######################## Get a dictionary of classes
def getClasses():
    classes = []
    with open("data/classes.names", "r") as f:
        for i, line in enumerate(f.readlines()):
            classes[i] = line.strip()
    return classes


######################## Prepare dataset
def prepareDataset():
    imgs = glob.glob('data/ts/ts/*.jpg')
    labs = glob.glob('data/ts/ts/*.txt')
    imgs = sorted(imgs, key= lambda x: x[11: 16]) # file_name 12345.jpg, begin index 11 in path
    labs = sorted(labs, key= lambda x: x[11: 16])
    N = len(imgs)
    X = np.zeros((N, IMG_SIZE, IMG_SIZE, DEPTH), dtype=np.float32)
    Y = np.zeros((N, NB_CELL, NB_CELL, 5+NB_CLASSES)) # p, x, y, w ,h, classes
    for i, (path_img,  path_lab) in enumerate(zip(imgs, labs)):
        #### IMAGE
        img = cv2.imread(path_img)
        # Simple preprocessing image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize
        #img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX) # normalize
        # normalize by dividing by 255
        X[i] = img/255.
        #### LABEL

        boxes = []
        classes = []

        with open(path_lab, 'r') as f:
            for line in f:
                line = line.strip().split(" ")
                boxes.append([float(x) for x in line[1:]]) # string to float
                classes.append(int(line[0]))
        
        boxes = np.array(boxes)
        for box, cl in zip(boxes, classes):
            # one hot coding
            cl_array = [0]*NB_CLASSES
            cl_array[cl] = 1
            # x, y, w, h are already scaled to img size [0-1]
            x, y, w, h = box[0]*IMG_SIZE, box[1]*IMG_SIZE, box[2], box[3] # rescale to true value of x, y
            inx_x, inx_y = int(x/CELL_SIZE), int(y/CELL_SIZE) # inx of cell is true coord divides by SIZE of cell
            x_center, y_center = (x-inx_x*CELL_SIZE)/CELL_SIZE, (y-inx_y*CELL_SIZE)/CELL_SIZE # coord relative to cell
            # make Y: 1 is I_object, we dont scale w, h because the box size is not relative to cell (can bigger then cell)
            # each cell can only detect 1 object so it can overwrite, need bigger number of cells to deal with this
            Y[i, inx_x, inx_y] = 1, x_center, y_center, w, h, *cl_array
    return X, Y


######################## Plot images and boxes
def show_IMG_BOXES(img_origin, lab, grid=True, from_pred=False):
    img = copy.deepcopy(img_origin) # avoid overwritting
    boxes = []
    if not from_pred:
        for i in range(NB_CELL):
            for j in range(NB_CELL):
                if lab[i, j, 0] == 1: # object
                    coord = lab[i, j, 1:5]
                    x_center, y_center, w, h = coord[0], coord[1], coord[2], coord[3]
                    x = x_center*CELL_SIZE+i*CELL_SIZE # reverse to true value [0-IMG_SIZE]
                    y = y_center*CELL_SIZE+j*CELL_SIZE # reverse to true value [0-IMG_SIZE]
                    w = w*IMG_SIZE
                    h = h*IMG_SIZE
                    top_left = (int(x-w/2), int(y-h/2))
                    bottom_right = (int(x+w/2), int(y+h/2))
                    upper_middle = (int(x), int(y-h/2)-5) # to put text, move a bit to the top by 5 pixel
                    classes = lab[i, j, -NB_CLASSES:]
                    cl = CLASSES[np.argmax(classes)]
                    boxes.append([cl, top_left, bottom_right, upper_middle])
    else:
      res = non_maximum_suppression(lab)
      for b in res:
        x1, y1, x2, y2, cl = b[1:]
        x_center, y_center  = (x1+x2)/2, (y1+y2)/2
        h = y2 - y1
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        upper_middle = (int(x_center), int(y_center-h/2)-5)
        cl = CLASSES[cl]
        boxes.append([cl, top_left, bottom_right, upper_middle])

    # plot box and class
    for box in boxes:
        img = cv2.rectangle(img, box[1], box[2], color=(255, 0, 0), thickness=2)
        img = cv2.putText(img, text=box[0], org=box[-1], fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                           fontScale= 0.5,\
                              color=(255, 0, 0),\
                                thickness=1)

    if grid:
        scale = IMG_SIZE//NB_CELL
        for i in range(scale):
            for j in range(scale):
                img = cv2.rectangle(img,\
                                    ((i * scale), (j * scale)), (((i+1) * scale),((j+1) * scale)),\
                                    color=(100,0,200), thickness=1)
    plt.imshow(img)


######################## Calculate IOU
def IoU(boxes1, boxes2):
    """
    Input
        boxes1: (Batch, Cell, Cell, 4) 4 are x, y, w, h
        boxes2: (Batch, Cell, Cell, 4) 4 are x, y, w, h

    Output:
        IoU : (Batch, Cell, Cell, 1)
    """
    # reform from (x, y, w, h) to (x1, y1, x2, y2)
    boxes1_reformed = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

    boxes2_reformed = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)
    top_left = tf.maximum(boxes1_reformed[..., 0:2], boxes2_reformed[..., 0:2])
    bottom_right = tf.minimum(boxes1_reformed[..., 2:4], boxes2_reformed[..., 2:4])

    # get width and height of intersection
    intersection_wh = tf.maximum(0.0, bottom_right - top_left)
    inter_square = intersection_wh[..., 0] * intersection_wh[..., 1]

    # calculate the boxs1 square and boxs2 square
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(area1 + area2 - inter_square, 1e-6)

    return tf.expand_dims(tf.clip_by_value(inter_square / union_square, 0.0, 1.0), axis=-1)    



######################## Loss function of Yolo algorithm
def loss_yolo(y, ypred):
    """
        ypred   : p_1, .., p_i, x_1, y_1, w_1, h_1, ..., x_i, y_i, w_i, h_i, c_1, ..., c_n
        y       : p, x, y, w, h, c_1, ... c_n
    """
    Iobj = y[..., 0] # (B, C, C)
    Iobj = tf.expand_dims(Iobj, axis=-1) # (B, C, C ,1)

    ### coord loss
    loss_coord = 0
    xy_true = y[..., 1:3] # (B, C, C, 2)
    wh_true = y[..., 3:5] # (B, C, C, 2)
    for i in range(NB_BOXES):
        current_box_position = NB_BOXES + i*4
        xy_pred = ypred[..., current_box_position: current_box_position+2] # (B, C, C, 2)
        wh_pred = ypred[..., current_box_position+2: current_box_position+4] # (B, C, C, 2)
        loss_coord += tf.reduce_sum(Iobj*tf.square(xy_true-xy_pred))
        loss_coord += tf.reduce_sum(Iobj*tf.square(tf.sqrt(wh_true)-tf.sqrt(wh_pred)))

    ### confidance loss
    loss_confidance = 0
    xywh_true = y[..., 1: 1+4] # (B, C, C, 4)
    for i in range(NB_BOXES):
        pred_obj_current_box = tf.expand_dims(ypred[..., i], axis=-1) # (B, C, C, 1)
        current_box_position = NB_BOXES + i*4
        xywh_pred = ypred[..., current_box_position: current_box_position+4] # (B, C, C, 4)
        iou = IoU(xywh_pred, xywh_true) # (B, C, C, 1)
        loss_confidance += tf.reduce_sum(Iobj*tf.square(pred_obj_current_box-iou))
        loss_confidance += ALPHA_NOOBJ*tf.reduce_sum((1-Iobj)*tf.square(pred_obj_current_box-iou))
    
    ### classification loss
    classe_pred = ypred[..., NB_BOXES*5: NB_BOXES*5+NB_CLASSES] # (B, C, C, nbClasses)
    classes_true = y[..., 5: 5+NB_CLASSES] # (B, C, C, nbClasses)
    loss_classification = tf.reduce_sum(Iobj*tf.square(classes_true-classe_pred))

    loss = ALPHA_COORD*loss_coord + loss_confidance + loss_classification
    return loss


def non_maximum_suppression(pred, IOU_threshold = 0.5, confidance_threshold = 0.4):
  """
    Interpretation the output to image scale values, then apply NMS algo
    Input:
      pred: 3-D Tensor (Cell, Cell, 5*nb_boxes + nb_classes) p_1, .., p_i, x_1, y_1, w_1, h_1, ..., x_i, y_i, w_i, h_i, c_1, ..., c_n
    Output:
      out: List of boxes whose coordinates are relative to IMAGE
  """
  def mini_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

    w_inter, h_inter = max(x2-x1, 0), max(y2-y1, 0)
    area_intersection = w_inter*h_inter

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box1[2]-box1[0], box1[3]-box1[1]
    area_1 = w1*h1
    area_2 = w2*h2

    union = area_1 + area_2 - area_intersection + 1e-6
    return area_intersection/union


  B = []
  for i in range(len(pred)):
    for j in range(len(pred[0])):
      cl = tf.math.argmax(pred[i, j, 5*NB_BOXES : 5*NB_BOXES+NB_CLASSES])
      for b in range(NB_BOXES):
        if pred[i, j, b] >= confidance_threshold:
          p = pred[i, j, b]
          x, y, w, h = pred[i, j, NB_BOXES+b*4 : NB_BOXES+b*4+4]
          x = x*CELL_SIZE+i*CELL_SIZE # reverse to true value [0-IMG_SIZE]
          y = y*CELL_SIZE+j*CELL_SIZE # reverse to true value [0-IMG_SIZE]
          w, h = w*IMG_SIZE, h*IMG_SIZE
          x1, y1 = x-w/2, y-h/2
          x2, y2 = x+w/2, y+h/2
          B.append([float(p), float(x1), float(y1), float(x2), float(y2), int(cl)])

  B = sorted(B, key=lambda x: x[0])[::-1]
  D = []
  while B != []:
    candidat = B.pop(0)
    D.append(candidat)
    inx_to_remove = []
    for i in range(len(B)):
      if mini_iou(candidat[1:5], B[i][1: 5]) >= IOU_threshold:
        inx_to_remove.append(i)
    B = [B[i] for i in range(len(B)) if i not in inx_to_remove]
  return D

