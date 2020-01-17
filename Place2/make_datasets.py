import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--root", default='/home/ouc/dataset/places365_hr/', help="root pass")
parser.add_argument("--dataset_name", type=str, default="Pic", help="name of the dataset")
args = parser.parse_args()
os.makedirs(args.dataset_name, exist_ok=True)

with open('./download_list.txt') as f:  # label name .txt
    w = f.read()
    w = w.split('\n')
    n = 0

    for i in w:
        if os.path.exists(args.root + 'data_large/' + i[0] + '/' + i):
            img_list = sorted(list(glob.glob(args.root + 'data_large/' + i[0] + '/' + i + "/*.*")))
#             output = os.path.join('./Pic',i)
#             if not os.path.isdir(output):
#                 os.mkdir(output)
#             print('./Pic/' + i)
            for j in range(len(img_list)):
                
                shutil.copy(img_list[j], './Pic/' + i +'/' + str(n) + '.png')
                n = n + 1
        else:
            print(args.root + i[0] + '/' + i + 'does not exist')
