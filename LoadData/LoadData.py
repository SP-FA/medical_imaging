import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class InputImg(Dataset):
    data = [];
    def __init__(self, data_dir, width, transform = None): #文件地址 './Data'
        classes = os.listdir(data_dir)
        k = 0
        for name in classes:
            name_ = os.listdir(data_dir+'/'+name)
            for two in name_:
                two_ = os.listdir(data_dir+'/'+name+'/'+two)
                for id in two_:
                    pictures = os.listdir(data_dir+'/'+name+'/'+two+'/'+id)
                    for pic in pictures:
                        img = Image.open(data_dir+'/'+name+'/'+two+'/'+id+'/'+pic)
                        if len(np.array(img).shape) == 2:
                            continue;
                        c1 = self.turn_img(np.array(img), width, 1)
                        c2 = self.turn_img(np.array(img), width, 2)
                        c3 = self.turn_img(np.array(img), width, 3)
                        if transform:
                            img = transform(img)
                            c1  = transform(c1)
                            c2  = transform(c2)
                            c3  = transform(c3)
                        self.data.append([img, k])
                        self.data.append([c1 , k])
                        self.data.append([c2 , k])
                        self.data.append([c3 , k])
            k += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def turn_img(self, img, width, case):
        c = Image.new("RGB", (width, width))
        for i in range(width):
            for j in range(width):
                w = width - i - 1
                h = width - j - 1
                if   case == 1:
                    rgb = tuple(img[j, w]) #镜像翻转 
                elif case == 2:
                    rgb = tuple(img[h, w]) #翻转180度  
                elif case == 3:
                    rgb = tuple(img[h, i]) #上下翻转  
                c.putpixel([i, j], rgb)
        return c