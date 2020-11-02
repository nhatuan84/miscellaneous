#convert labelme format to format below
#xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height


import json
import cv2
import numpy as np


img_path =   'pic.png'
json_path =  img_path.replace('.png', '.json')
img = cv2.imread(img_path)

f = open("train.txt", "w")

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
        cv2.imshow('', roi)
        cv2.waitKey(0)
    f.write(line)
    f.write('\n')
f.close()
