# !/usr/bin/env python
# -*- coding:utf-8 -*-
from mmdet.apis import init_detector, inference_detector, show_result
import os
import json
import numpy as np
import glob
import time

config_file = 'configs/cascade_rcnn_x101_64x4d_fpn_1x_round2.py'

checkpoint_file = 'docker/epoch_30.pth'


# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
class_list = ['沾污', '错花', '水印', '花毛', '缝头', '缝头印', '虫粘',
              '破洞', '褶子', '织疵', '漏印', '蜡斑', '色差', '网折', '其他']
real_class = {
    '无疵点': 0, '沾污': '1', '错花': '2', '水印': '3', '花毛': '4', '缝头': '5', '缝头印': '6', '虫粘': '7',
    '破洞': '8', '褶子': '9', '织疵': '10', '漏印': '11', '蜡斑': '12', '色差': '13', '网折': '14', '其他': '15'}


# 测试一系列图片
test_img_path = '/tcdata/guangdong1_round2_testA_20190924/'
dir_names = []
for root, dirs, files in os.walk(src_img_dir): 
    for dir in dirs: 
        dir_names.append(dir)
images_name = []
for _ in dir_names:
    images_name.append(test_img_path + _ + '/' + _ + '.jpg')

score_threshold = 0.00
res = []
image_name = []
defect_label = []
bbox = []
score_res = []
for i in range(len(images_name)):
	result = inference_detector(model, images_name[i])
	#print('img', i)
	label, box = show_result(images_name[i], result, model.CLASSES)
	for j in range(len(label)):
		temp = box[j]
		score = temp[4]
		if score > score_threshold:
			image_name.append(images[i])
			defect_label.append(real_class[class_list[label[j]]])
			score_res.append(float(score))
			x_min = temp[0]
			y_min = temp[1]
			x_max = temp[2]
			y_max = temp[3]
			x_min = round(float(x_min), 2)
			y_min = round(float(y_min), 2)
			x_max = round(float(x_max), 2)
			y_max = round(float(y_max), 2)
			bbox.append([x_min, y_min, x_max, y_max])
for i in range(len(image_name)):
    res.append({'name': image_name[i], 'category': defect_label[i], 'bbox': bbox[i], 'score': score_res[i]})

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

with open('result_' + now + '.json', 'w') as fp:
    json.dump(res, fp, indent=4, separators=(',', ': '))

