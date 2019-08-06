from __future__ import print_function
import os
import cv2
import numpy as np
from . import selectivesearch
import matplotlib.pyplot as plt

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    '''
    
    :param in_image: input image
    :param new_width: width of the new image
    :param new_height: height of the new image
    :param out_image: new path of the new image
    :param resize_mode: resize mode in CV2
    :return: the new image
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

def clip_pic(img, rect):
    '''
    
    :param img: input image
    :param rect: four params of the rect
    :return: the clipped image and the rect
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]

def IOU(ver1, vertice2):
    '''
    calculate the IOU of two verts
    :param ver1: the first vert
    :param vertice2: the second vert
    :return: IOU value
    '''
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter
    
def image_proposal(img_path, img_size):
    '''
    use selectivesearch to obtain image proposals;
    add some rules to refine these image proposals
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    rects = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 200:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        proposal_img, proposal_vertice = clip_pic(img, r['rect'])
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize_image(proposal_img, img_size, img_size)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(proposal_vertice)
        rects.append(r['rect'])
    return images, vertices, rects

def show_rect(img_path, regions, message):
    '''
    :param img_path: input image
    :param regions: rects
    :param message: annotation
    :return: 
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y, w, h in regions:
        x, y, w, h =int(x),int(y),int(w),int(h)
        rect = cv2.rectangle(
            img,(x, y), (x+w, y+h), (0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message, (x+20, y+40),font, 1,(255,0,0),2)
    plt.imshow(img)
    plt.show()
