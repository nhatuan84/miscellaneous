# object-detection-utils
object detection utils

1.==============================================================================
https://github.com/nhatuan84/object-detection-utils/blob/main/convert-labelme.py

Convert labelme format to format below:

image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 

Note: make sure that x_max < width and y_max < height
(e.g: xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20)

===============================================================================
