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


class SIATFolder(data.Dataset):

    def __init__(self, root, index, transform, istrain, config):
        self.istrain = istrain
        self.config = config
        self.data = []
        self.img_path = root + "distortion2D"
        self.pc_path = root + "siat_6patch_2048"

        mos_index = ['Longdress', 'Andrew', 'UlliWegner', 'RedAndBlack', 'Ricardo', 'The20sMaria', 'Phil', 'Loot',
                     'Sarah', 'Soldier', 'Grass', 'Biplane', 'Banana', 'AngelSeated', 'RomanOillamp', 'Facade', 'Bush',
                     'House', "Nike", 'ULBUnicorn']
        split_index = ['Andrew', 'AngelSeated', 'Banana', 'Biplane', 'Bush', 'Facade', 'Grass', 'House', 'Longdress', 'Loot',
                 'Nike', 'Phil', 'RedAndBlack', 'Ricardo', 'RomanOillamp', 'Sarah', 'Soldier', 'The20sMaria',
                 'ULBUnicorn', 'UlliWegner']

        if istrain:
            index_order = split_index[:index * 4 - 4] + split_index[index * 4:]
        else:
            index_order = split_index[index * 4 - 4:index * 4]

        mos = pd.read_excel(os.path.join(root, 'DMOS.xlsx'), usecols=[5])

        use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        a_path = root + "spatial_position.mat"
        A = scio.loadmat(a_path)["A"]

        for file_type in index_order:
            files = os.listdir(os.path.join(self.img_path, file_type))
            for f in files:
                file_list = []
                num = f.split(".ply")[0].split('_r')[1]
                file_name = file_type + '_r' + num
                for j in use_number:
                    file_name2 = file_name + "_" + str(j) + ".png"
                    file_path = os.path.join(self.img_path, file_type, file_name2)
                    file_list.append(file_path)

                ply_name = file_name + ".npy"
                ply_path = os.path.join(self.pc_path, ply_name)

                mos_ind = mos_index.index(file_type)
                label = float(mos.iloc[mos_ind*17+int(num)-1])

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
