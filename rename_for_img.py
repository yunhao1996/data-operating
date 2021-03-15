import os
import cv2 as cv

image_path = './img/'
rename_path = 'rename_img/'

for file in os.listdir(image_path):
    new_file = file.replace('_s_', '_')
    print(new_file)
    new_image = rename_path + new_file
    print(new_image)


    scr = cv.imread(image_path + file)
    cv.imwrite(new_image, scr)
