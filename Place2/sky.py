import os
from PIL import Image

val_path = '/home/ouc/dataset/places365_hr/places365_val.txt'
path = '/home/ouc/dataset/places365_hr/val_large/'
output = '/home/ouc/dataset/places365_hr/desert_root_val'
img = []
if not os.path.isdir(output):
    os.mkdir(output)

for i in range(36500):
    with open(val_path, 'r') as f1:
        line_f1 = f1.readlines()
    line_f1[i] = line_f1[i][:-1]
    
    label = line_f1[i].split(' ')[0]
    category = line_f1[i].split(' ')[1]

    if category=='118':
        img_path = path+line_f1[i].split(' ')[0]
        print(img_path)
        img = Image.open(img_path)
        img.save(output+"/"+line_f1[i].split(' ')[0])
        
           
         
    
   
