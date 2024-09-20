import torch.utils.data as data
import os
import scipy.io as scio
import torch
import numpy as np
from PIL import Image, ImagePath
import pandas as pd


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class WPCFolder(data.Dataset):

    def __init__(self, root, index, transform, istrain, config):
        self.istrain = istrain
        self.config = config

        order = ['bag', 'banana', 'biscuits', 'cake', 'cauliflower', 'flowerpot', 'glasses_case',
                 'honeydew_melon', 'house', 'litchi', 'mushroom', 'pen_container', 'pineapple',
                 'ping-pong_bat', 'puer_tea', 'pumpkin', 'ship', 'statue', 'stone', 'tool_box']

        # Split training and test images based on indices
        split_index = ['banana', 'cauliflower', 'mushroom', 'pineapple', 'bag', 'biscuits', 'cake',
                       'flowerpot', 'glasses_case', 'honeydew_melon', 'house', 'pumpkin', 'litchi',
                       'pen_container', 'ping-pong_bat', 'puer_tea', 'ship', 'statue', 'stone', 'tool_box']
        if istrain:
            index_order = split_index[:index * 4 - 4] + split_index[index * 4:]
        else:
            index_order = split_index[index * 4 - 4:index * 4]

        # Extracting labels
        mos = pd.read_excel(os.path.join(root, 'WPC_MOS.xlsx'))

        self.data = []
        self.img_path = root + "distorted2D"
        self.pc_path = root + "wpc_6patch_2048"

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        a_path = os.path.join(root + "spatial_position.mat")
        A = scio.loadmat(a_path)["A"]

        for i in range(len(mos)):
            ind = None
            file_list = []
            file_name = mos.iloc[i, 1].split(".ply")[0]
            if "_pqs_1_qs" in file_name:
                file_name = file_name + "_rounded"

            for m in index_order:
                if m in file_name:
                    ind = m
                    break
            if ind is None:
                continue

            for j in use_number:
                file_name2 = file_name + "_" + str(j) + ".png"
                file_path = os.path.join(self.img_path, ind, file_name2)
                file_list.append(file_path)

            ply_name = file_name + ".npy"
            ply_path = os.path.join(self.pc_path, ply_name)

            label = mos.iloc[i, 2] / 10
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
