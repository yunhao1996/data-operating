import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset', default=r'D:\软件2\test\A')
parser.add_argument('--output', type=str, help='path to the TXT list', default=r'D:\软件2\1.txt')
args = parser.parse_args()

ext = {'.jpg', '.png', 'JPG', 'PNG'}

images = []
for root, dirs, files in os.walk(args.path):
    # print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(root.replace(args.path, 'A'), file) + ' ' + os.path.join(root.replace(args.path, 'B'),  file))

print(images)

np.savetxt(args.output, images, fmt='%s')