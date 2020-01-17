import os
from PIL import Image

PIC_size = 256

path = './test.txt'  # the flist path
output = '/home/ouc/dataset/Cars_dataset/car-sort'  # output path
img = []
if not os.path.isdir(output):
    os.mkdir(output)

j=8144  # the number of image
for i in range(8041):
    with open(path, 'r') as f:
        line_f = f.readlines()

    line_f[i] = line_f[i][:-1]
    img = Image.open(line_f[i])
#     img_r = img.resize((PIC_size, PIC_size), Image.ANTIALIAS)
    j +=1
    img.save(output+"/%05d" % j + ".jpg")
    print(i)
