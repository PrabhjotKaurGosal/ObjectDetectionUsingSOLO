#!/usr/bin/env python
# coding: utf-8

# In[213]:


# This file converts object detection annotations from YOLO format (.txt and .names) to COCO format (.json)
# The base code is originally from: https://www.programmersought.com/article/76707275021/
# It was modified as the original code was not working for my needs.


# In[224]:


import os
import json
import cv2
import random
import time


# In[129]:


# Preparing the Dataset
# Convert all images to .jpg (This is not necessary but made my life easier down the line)

from PIL import Image
#iml = Image.open(r'/media/jyoti/My Passport/DoorDetectDataset/images/Wood_Mode_Brookhaven_Hardware_Kitchen_Associates-3.png')
#rgb_im = iml.convert('RGB')
#rgb_im.save(r'/media/jyoti/My Passport/DoorDetectDataset/images/Wood_Mode_Brookhaven_Hardware_Kitchen_Associates-3.jpg')


# In[221]:


# Preparing the Dataset - remove unlabelled files
# Move the files (jpg and txt that are not labelled) from the working directory and put them in another folder. . .
# I chose to remove the image files from the dataset that were not labelled

import shutil

directory_labels = os.fsencode("/media/jyoti/My Passport/DoorDetectDataset/labels") #absolute path to the directory where all labels are stored
directory_images = os.fsencode("/media/jyoti/My Passport/DoorDetectDataset/images") #absolute path to the directory where all images are stored
directory_unlabelled_images = os.fsencode("/media/jyoti/My Passport/DoorDetectDataset/Unlabelled_images")#absolute path to the directory where unlabelled images are to be stored

not_labelled_files = 0
for file in os.listdir(directory_labels): #Read from the directory where the labels are stored
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
        yolo_annotation_path = (os.path.join(directory_labels.decode("utf-8"), filename))
        base=os.path.basename(yolo_annotation_path)
        file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
        img_path = os.path.join(directory_images.decode("utf-8"), file_name_without_ext+ "." + 'jpg')
        
        filesize = os.path.getsize(yolo_annotation_path)
        if filesize == 0: #chek if the label file (.txt) is empty - that is it has no labels
            UnlabeledImg_path = os.path.join(directory_unlabelled_images.decode("utf-8"), file_name_without_ext+ "." + 'jpg')
            not_labelled_files = not_labelled_files +1
            
            shutil.move(img_path, UnlabeledImg_path) #remove the corresponding image to another directory
print("The number of unlabelled files is: ", not_labelled_files)


# In[225]:


# coco format last storage location
coco_format_save_path = '/media/jyoti/My Passport/DoorDetectDataset/0a604ed4883b45e5.json'
 # Category file, one category per line
yolo_format_classes_path = '/media/jyoti/My Passport/DoorDetectDataset/obj.names'
 # yolo format comment file
#yolo_format_annotation_path = '/media/jyoti/My Passport/DoorDetectDataset/labels/0a604ed4883b45e5.txt'
 # Write the category according to your own data set. for example: 
# categories_dict = [{'supercategory': 'None', 'id': 1, 'name': 'w3'},{'supercategory': 'None', 'id': 2, 'name': '

#Read the categories file and extarct all categories
with open(yolo_format_classes_path,'r') as f1:
    lines1 = f1.readlines()
categories = []
for j,label in enumerate(lines1):
    label = label.strip()
    categories.append({'id':j+1,'name':label,'supercategory': 'None'})
    
write_json_context = dict()
write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021-02-12 11:00:08.5'}
write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = categories
write_json_context['images'] = []
write_json_context['annotations'] = []


# In[226]:


#Read the label files (.txt) to extarct bounding box information and store in COCO format
directory_labels = os.fsencode("/media/jyoti/My Passport/DoorDetectDataset/labels")
directory_images = os.fsencode("/media/jyoti/My Passport/DoorDetectDataset/images")

file_number = 1
num_bboxes = 1
for file in os.listdir(directory_labels):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
        yolo_annotation_path = (os.path.join(directory_labels.decode("utf-8"), filename))
        base=os.path.basename(yolo_annotation_path)
        file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
        img_path = os.path.join(directory_images.decode("utf-8"), file_name_without_ext+ "." + 'jpg')
        img_name = os.path.basename(img_path) # name of the file without the extension
        img_context = {}
        height,width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = img_name
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2021-02-12 11:00:08.5'
        img_context['id'] = file_number # image id
        img_context['license'] = 1
        img_context['coco_url'] =''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)
        
        with open(yolo_annotation_path,'r') as f2:
            lines2 = f2.readlines() 

        for i,line in enumerate(lines2): # for loop runs for number of annotations labelled in an image
            line = line.split(' ')
            bbox_dict = {}
            class_id, xmin,ymin,xmax,ymax,= line[0:]
            xmin,ymin,xmax,ymax,class_id= float(xmin),float(ymin),float(xmax),float(ymax),int(class_id)
            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_number
            bbox_dict['category_id'] = class_id + 1
            bbox_dict['iscrowd'] = 0 # There is an explanation before
            h,w = abs(ymax-ymin),abs(xmax-xmin)
            bbox_dict['area']  = h * w
            bbox_dict['bbox'] = [xmin,ymin,w,h]
            bbox_dict['segmentation'] = [[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
            write_json_context['annotations'].append(bbox_dict)
            num_bboxes+=1
        
        file_number = file_number+1
        continue
     else:
        continue
        
 # Finally done, save!
coco_format_save_path = '/media/jyoti/My Passport/DoorDetectDataset/COCO_format_labels.json'
with open(coco_format_save_path,'w') as fw:
    json.dump(write_json_context,fw)          
    


# In[ ]:




