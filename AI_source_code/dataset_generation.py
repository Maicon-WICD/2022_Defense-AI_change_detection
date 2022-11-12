import os, json, glob, shutil
from PIL import Image
from shapely import geometry
import PIL.ImageDraw as ImageDraw
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def read_json_file(json_path):
    with open(json_path, encoding='utf-8-sig') as f:
        data = json.load(f)
    return data

"""Read json & image files"""
# <change the root path>
root_dir = "C:/Dacon/Wonguk/2022_Defense-AI_change_detection/train/"

# <set the path to json and image files>
# json_file_path = root_dir + "json/*.json"                         # 주석처리
image_file_path = root_dir + 'x/*.png'

# json_files = glob.glob(json_file_path)                            # 주석처리
image_files  = glob.glob(image_file_path, recursive=True)
print(len(image_files))
# Extract polygons from json files
# print("Read polygons...")
# total_polygons = []
dataset = []
for i in range(len(image_files)):   
    image_name = os.path.basename(image_files[i])
#     # json_file = [ j for j in json_files if image_name in j]       # 주석처리
#     # if len(json_file) == 0:
#     #     dataset.append({"image-name":  image_files[i],'polygons':[],'points':[]})
#     #     continue
#     # json_file = json_file[0]
#     # json_data = read_json_file(json_file)
    each_image_data = []
#     # polygons = []
#     # points = []   # 주석처리 끝
    each_image_data = {"image-name": image_files[i]}  
#     for annotation in json_data['annotations']:
#         polygons.append(geometry.Polygon(annotation['polygon.points']))
#         points.append(annotation['polygon.points'])
#         total_polygons.append(geometry.Polygon(annotation['polygon.points']))
    
#     each_image_data['polygons'] = polygons
#     each_image_data['points'] = points
    dataset.append(each_image_data)

# Generate dataset----------------------------------------------------------------------
print("Dataset generation....")
# Create 3 folders for generated dataset: 'input1', 'input2' and 'mask'     # 주석
input1_dir = root_dir + 'input1'
input2_dir = root_dir + 'input2'
mask_dir = root_dir + 'mask'
if not os.path.exists(input1_dir):
    os.makedirs(input1_dir)
if not os.path.exists(input2_dir):
    os.makedirs(input2_dir)
# if not os.path.exists(mask_dir):              # 주석
#     os.makedirs(mask_dir)    

save_path = root_dir + "%s/%s"
# mask_path = root_dir + "mask/%s"              # 주석
print(1)
cnt = 1
for data_item in dataset:
    image = Image.open(data_item['image-name'])
    image = np.asarray(image)
    img1 = Image.fromarray(image[:,:754,:])
    img2 = Image.fromarray(image[:,754:,:])
    print('image save count = ' ,cnt)
    img1.save(save_path%("input1",(os.path.basename(data_item['image-name']))))
    img2.save(save_path%("input2",(os.path.basename(data_item['image-name']))))
    # img1.save(save_path%("input1",(os.path.basename(data_item['image-name'])).replace(".tif",".png")))
    # img2.save(save_path%("input2",(os.path.basename(data_item['image-name'])).replace(".tif",".png")))
    mask_image = Image.fromarray(np.zeros_like(np.asarray(image)))
    cnt += 1

print('End')

    # if len(data_item['points']) !=0:                      # 주석처리
    #     draw = ImageDraw.Draw(mask_image)
    #     for each_poly_points in data_item['points']: 
    #         if len(each_poly_points) == 0: continue
    #         points = (tuple([tuple(point) for point in each_poly_points]))
    #         points = (*points ,points[0])
    #         draw.polygon((points), fill='white')
    
    # mask_image = np.asarray(mask_image)
    # img1 =  mask_image[:,:754,:]
    # img2 =  mask_image[:,754:,:]
    # mask_image = Image.fromarray(img1 + img2)
    # mask_image.save(mask_path%(os.path.basename(data_item['image-name'])).replace(".tif",".png"))


# # Split the dataset into train and set with ratio: 0.8:0.2 -----------------------------------------------
# data_imgList = os.listdir(root_dir + 'input1')

# # data_train, data_val = train_test_split(data_imgList, test_size=0.2, random_state=42)

# # """copy imgs into train & test folder""" ---------------
# in_path = root_dir
# out_path = root_dir + 'train'
# os.makedirs(out_path)
# os.makedirs(out_path+'/input1')
# os.makedirs(out_path+'/input2')
# os.makedirs(out_path+'/mask')
# shutil.copytree(out_path, root_dir + 'val')

# for im_id in data_train:
#     shutil.copy(in_path + '/input1/' + im_id, out_path + '/input1/')
#     shutil.copy(in_path + '/input2/' + im_id, out_path + '/input2/')
#     shutil.copy(in_path + '/mask/' + im_id, out_path + '/mask/')
    
# out_path = root_dir + 'val'
# for im_id in data_val:
#     shutil.copy(in_path + '/input1/' + im_id, out_path + '/input1/')
#     shutil.copy(in_path + '/input2/' + im_id, out_path + '/input2/')
#     shutil.copy(in_path + '/mask/' + im_id, out_path + '/mask/')
