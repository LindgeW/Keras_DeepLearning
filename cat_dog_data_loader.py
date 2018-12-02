from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
import os

def load_data(img_dir, img_size):
    files = os.listdir(img_dir)
    images = []
    labels = []
    for f in files:
        img_path = os.path.join(img_dir, f)
        img = image.load_img(img_path, target_size=img_size) #img_size = (img_w,img_h)
        img_array = image.img_to_array(img)
        images.append(img_array)

        #打标签：cat - 0   dog - 1
        if 'cat' in f: #文件名中包含cat
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels