import os
from PIL import Image
import numpy as np
from tqdm import tqdm
folder_path = './data/train/y/'
count1 = []
count2 = []
count3 = []
f1 = open("label_1_list.txt", 'w')
f2 = open("label_2_list.txt", 'w')
f3 = open("label_3_list.txt", 'w')
data_list = os.listdir(folder_path)
for k in tqdm(data_list):
    im = Image.open(folder_path + k)
    pix = np.array(im).tolist()
    sum = []
    for i in pix:
        if 1 in i:
            f1.write(folder_path + k+'\n')
            count1.append(1)
            break
        elif 2 in i:
            f2.write(folder_path + k+'\n')
            count2.append(2)
            break
        elif 3 in i:
            f3.write(folder_path + k+'\n')
            count3.append(3)
            break

f1.close()
f2.close()
f3.close()
print(len(count1),len(count2),len(count3))