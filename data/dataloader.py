import torch
import torchvision
import data.folder


class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, config, dataset, path, split, image_size, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        img_resize = 448

        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((img_resize, img_resize)),
                torchvision.transforms.CenterCrop((image_size, image_size)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((img_resize, img_resize)),
                torchvision.transforms.CenterCrop((image_size, image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

        if dataset == "sjtu":
            self.data = data.folder.SJTUFolder(root=path, index=split, transform=transforms,
                                               istrain=istrain, config=config)
        elif dataset == "wpc":
            self.data = data.folder.WPCFolder(root=path, index=split, transform=transforms,
                                              istrain=istrain, config=config)
        elif dataset == "siat":
            self.data = data.folder.SIATFolder(root=path, index=split, transform=transforms,
                                               istrain=istrain, config=config)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
