import json
import cv2
import numpy as np
import os
import random


def convert_from_labelme(img_path, output="train.txt"):
    json_path =  img_path.replace('.jpg', '.json')
    img = cv2.imread(img_path)
    f = open(output, "w")
    with open(json_path) as json_file:
        data = json.load(json_file)
        keypoints = data['shapes']
        class_id = 1
        line = img_path
        for kps in keypoints:
            pt_list = kps['points']
            pt_list = sorted(pt_list, key=lambda x:x[0])
            xmin = pt_list[0][0]
            xmax = pt_list[-1][0]
            pt_list = sorted(pt_list, key=lambda x:x[1])
            ymin = pt_list[0][1]
            ymax = pt_list[-1][1]
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2) 
            sub_line =  ' ' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(class_id)
            line += sub_line
            roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        f.write(line)
        f.write('\n')
    f.close()


def image_bbox_resize(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def parse_annotation(annotation_line):
        line = annotation_line.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        return image_path, bboxes


#image_path, bboxes = parse_annotation('img.jpg 72.48576850094877,49.49525616698292,196.01518026565464,119.32447817836811,1')
#image_paded, gt_boxes = image_bbox_resize(cv2.imread(image_path), (416,416), bboxes)
#bboxes = bboxes.tolist()
#xmin = bboxes[0][0]
#ymin = bboxes[0][1]
#xmax = bboxes[0][2]
#ymax = bboxes[0][3]
#cv2.rectangle(image_paded, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2) 
#cv2.imwrite('image_paded.jpg', image_paded)


#boxes is in format (xmin, ymin, xmax, ymax)
def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

#boxes1 = [5, 4, 10, 10]
#boxes2 = [7, 7, 15, 15]
#iou = bboxes_iou(boxes1, boxes2)
#print(iou)


def preprocess_true_boxes(bboxes):
    train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    train_input_size = random.choice(train_input_sizes)
    strides = np.array([8, 16, 32])
    train_output_sizes = train_input_size // strides
    max_bbox_per_scale = 150
    anchor_per_scale = 3
    num_classes = 9
    anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = np.array(anchors.split(','), dtype=np.float32)
    anchors = anchors.reshape(3, 3, 2)


    label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale, 5 + num_classes)) for i in range(3)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    #scale ground-truth (center, w, h)
    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = bbox[4]

        #make onehot less confident
        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
        
        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]
            #just care gt iou achor > 0.3
            iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                #center of gt
                xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                #label = 
                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

bboxes = [np.array([72, 49, 196, 119, 1])]
preprocess_true_boxes(bboxes)
