import os
from PIL import Image

val_path = '/home/ouc/dataset/places365_hr/places365_val.txt'
categories_path = '/home/ouc/dataset/places365_hr/categories_places365.txt'
output = '/home/ouc/dataset/places365_hr/val'
img = []
if not os.path.isdir(output):
    os.mkdir(output)


with open(categories_path, 'r') as f2:
    line_f2 = f2.read()
    line_f2 = line_f2.split('\n')
    print(line_f2[0][-1]) 
for i in range(1):
    print(i)
    with open(val_path, 'r') as f1:
        line_f1 = f1.readlines()
    line_f1[i] = line_f1[i][:-1]
    
    label = line_f1[5].split(' ')[0]
    category = line_f1[5].split(' ')[1]
    for j in range(365):
        if category == line_f2[j][-1]:
            break
         
        category = line_f2[j][:-2]
    img_path = '/home/ouc/dataset/places365_hr/val_large/'+ label
    print(img_path)
    print(label)
    print(category)
    
    
#     print(line_f[i])
#     img = Image.open(line_f[i])
# #     img_r = img.resize((PIC_size, PIC_size), Image.ANTIALIAS)
#     j +=1
#     img.save(output+"/%05d" % j + ".jpg")
#     print(i)