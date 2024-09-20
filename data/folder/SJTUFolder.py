import torch.utils.data as data
import os
import scipy.io
import scipy.io as scio
import torch
import numpy as np
import random
from PIL import Image, ImagePath


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SJTUFolder(data.Dataset):

    def __init__(self, root, index, transform, istrain, config):
        self.istrain = istrain
        self.config = config

        order = ['redandblack', 'Romanoillamp', 'loot', 'soldier', 'ULB Unicorn', 'longdress', 'statue', 'shiva', 'hhi']

        # Split training and test images based on indices
        if istrain:
            index_order = order[:index - 1] + order[index:]
        else:
            index_order = order[index - 1:index]

        # Extracting labels
        mos = scipy.io.loadmat(os.path.join(root, 'Final_MOS.mat'))
        labels = mos['Final_MOS'].astype(np.float32)

        self.data = []
        self.img_path = root + "distortion2D"
        self.pc_path = root + "sjtu_6patch_2048"

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        a_path = os.path.join(root + "spatial_position.mat")
        A = scio.loadmat(a_path)["A"]

        for ind in index_order:
            for img_num in range(42):
                file_list = []
                for j in use_number:
                    file_name = ind + "_" + str(img_num) + "_" + str(j) + ".png"
                    file_path = os.path.join(self.img_path, ind, file_name)
                    file_list.append(file_path)

                ply_name = ind + "_" + str(img_num) + ".npy"
                ply_path = os.path.join(self.pc_path, ply_name)

                order_ind = order.index(ind)
                label = labels[img_num][order_ind]
                self.data.append((file_list, label, A, ply_path))

        self.transform = transform
        self.patch_length_read = 6
        self.npoint = 2048
        print("load dataset num:", len(self.data))

    def __getitem__(self, index):

        file_list, label, A, ply_path = self.data[index]

        imgs = None
        for i in file_list:
            if imgs is None:
                imgs = self.transform(pil_loader(i)).unsqueeze(0)
            else:
                file = self.transform(pil_loader(i)).unsqueeze(0)
                imgs = torch.cat((imgs, file), dim=0)

        selected_patches = torch.zeros([self.patch_length_read, 3, self.npoint])
        points = list(np.load(ply_path))

        for i in range(self.patch_length_read):
            selected_patches[i] = torch.from_numpy(points[i]).transpose(0, 1)
        return imgs, label, A, selected_patches

    def __len__(self):
        length = len(self.data)
        return length
