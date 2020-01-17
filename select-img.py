import os
from PIL import Image

# path: score txt;  root: img folder;  output: save image
path = '/home/ouc/Codes/2020pic/inn/celeba/psnrnew.txt'
root = '/home/ouc/Codes/2020pic/inn/celeba/'
output = '/home/ouc/Codes/2020pic/select-inn/celeb/'
k = 0

if (not os.path.isdir(output)):
    os.mkdir(output)
    
label = ['gt','my','other']  # folder name
for label1 in label:
    print(label1)

    for i in range(1222):
        with open(path, 'r') as f:
            line_f = f.readlines()

        line_f[i] = line_f[i][:-1]
        score = line_f[i].split(' ')[0]

        if float(score)>5:
            img_ext = line_f[i].split(' ')[1]
            os.path.splitext(img_ext)
            img = Image.open(os.path.join(root, label1, img_ext))
            
            number = os.path.splitext(img_ext)[0]
            img.save(output + '%d' % int(number) + label1 +'.jpg')
            print(i)
