# ⭐DHCN
Official repo for "Dynamic Hypergraph Convolutional Network for No-Reference Point Cloud Quality Assessment", accepted by TCSVT.

## 📋Environment
python 3.7  
pytorch 1.13.1  
pytorch-cuda 11.6  
sklearn  
cvxpy 1.3.1  
tqdm  
pandas  
open3d  
skimage  
 
## 📖Training
We train the code on the Ubuntu 18.04 system, the GPU is 3090 with 24 GB memory.
You can simply train DHCN with the following command:
```
python train.py
```

We provide the download links for the projections and patches of the WPC dataset, which can be accessed here ([Onedrive]()).  
The file structure is like:
```
WPC
├── distorted2D
│   ├── bag
│   │   ├── bag_gQP_1_tQP_1_1.png
│   │   ├── bag_gQP_1_tQP_1_2.png
│   │   ├── bag_gQP_1_tQP_1_3.png
...
├── wpc_6patch_2048
│   ├── bag_gQP_1_tQP_1.npy
│   ├── bag_gQP_1_tQP_2.npy
│   ├── bag_gQP_1_tQP_3.npy
...
```
If you want to process the dataset yourself, you can run 'utils/get_patch.py' and 'utils/get_projection2D.py'.

## 📖Testing
You can simply test DHCN with the following command:
```
python test.py
```
We provide the pre-trained weights on the WPC database ([Onedrive]()).
They are obtained by k-fold cross validation, and the final performance result is the average results over all test sets.

## 🔍Citation
If you find our work useful, please give us star and cite our paper as:
```
article{chen2024dhcn,
  title={Dynamic Hypergraph Convolutional Network for No-Reference Point Cloud Quality Assessment}, 
  author={Chen, Wu and Jiang, Qiuping and Zhou, Wei and Xu, Long and Lin, Weisi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2024},
}
```
