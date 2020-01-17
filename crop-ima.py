import os
from PIL import Image
import cv2
import numpy as np

# pt = '/home/ouc/dataset/dtd-r1.0.1/dtd/images'
path = '/home/ouc/train.flist'
output = '/home/ouc/train'
# flist_path = '/home/ouc/dataset/dtd-r1.0.1/dtd/train.flist'

# if (not os.path.isdir(output)):
#     os.mkdir(output)
# images = []
for i in range(5000):
    print(i)
    with open(path, 'r') as f:
        line_f = f.readlines()
    
    line_f[i] = line_f[i][:-1]
    a = line_f[i].split('/')[-1]  

#     complete_path = os.path.join(pt,line_f[i])  # splice the path
    img = cv2.imread(line_f[i])
    
# suggest the way to resize image
    img_c = cv2.resize(img, dsize=(512,256), interpolation=cv2.INTER_AREA)  
    img_path = os.path.join(output, a)
    cv2.imwrite(img_path, img_c)

# other way to resize and save image
# images.append(img_path)
# np.savetxt(flist_path, images, fmt='%s')

