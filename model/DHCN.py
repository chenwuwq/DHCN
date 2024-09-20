import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from sklearn.metrics import pairwise_distances
import cvxpy as cp
from cvxpy.error import SolverError
import scipy.sparse as sparse
from model.PcNet import PCNet

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Attention_Block(nn.Module):
    def __init__(self, input_channels):
        super(Attention_Block, self).__init__()

        self.Conv1x1 = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(input_channels, input_channels // 16, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(input_channels // 16, input_channels, kernel_size=1, bias=False)

    def forward(self, x):
        res = x

        x1 = self.Conv1x1(x)
        att1 = self.sigmoid(x1)

        x2 = self.avgpool(x)
        x2 = self.Conv_Squeeze(x2)
        x2 = self.Conv_Excitation(x2)
        att2 = self.sigmoid(x2)

        out = (att1 * att2) * res + res
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN_layer(nn.Module):
    def __init__(self, img_len, in_c):
        super(HGCN_layer, self).__init__()
        self.gc1 = GraphConvolution(in_c, in_c)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc2 = GraphConvolution(in_c, in_c)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc3 = GraphConvolution(in_c, in_c)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(feature + gc1)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(feature + gc2)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(feature + gc3)
        return gc3


class HGCNNet(nn.Module):
    def __init__(self, img_len):
        super(HGCNNet, self).__init__()
        self.gc1 = GraphConvolution(1024, 512)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer1 = HGCN_layer(img_len, 512)

        self.gc2 = GraphConvolution(512, 256)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer2 = HGCN_layer(img_len, 256)

        self.gc3 = GraphConvolution(256, 128)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer3 = HGCN_layer(img_len, 128)

        self.gc4 = GraphConvolution(128, 32)
        self.bn4 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer4 = HGCN_layer(img_len, 32)

        self.gc5 = GraphConvolution(32, 1)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc1 = self.HGCN_layer1(gc1, H)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc2 = self.HGCN_layer2(gc2, H)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)
        gc3 = self.HGCN_layer3(gc3, H)

        gc4 = self.gc4(gc3, H)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)
        gc4 = self.HGCN_layer4(gc4, H)

        gc5 = self.gc5(gc4, H)
        gc5 = self.relu(gc5)
        return gc5


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DHCN(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(DHCN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.HGCN_Attention1 = Attention_Block(256)
        self.HGCN_Attention2 = Attention_Block(512)
        self.HGCN_Attention3 = Attention_Block(1024)
        self.HGCN_Attention4 = Attention_Block(2048)
        self.HGCNlinear1 = nn.Linear(512, 256)
        self.HGCNlinear2 = nn.Linear(1024, 256)
        self.HGCNlinear3 = nn.Linear(2048, 256)

        img_len = 10
        self.HGCN_W = Parameter(torch.ones(img_len * 4))
        self.HGCN = HGCNNet(img_len=img_len)

        self.HGCN_linear6 = nn.Linear(2, 1)
        self.pcnet = PCNet()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def feature_exact(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = self.HGCN_Attention1(x1)
        x2 = self.HGCN_Attention2(x2)
        x3 = self.HGCN_Attention3(x3)
        x4 = self.HGCN_Attention4(x4)
        g1 = self.avgpool(x1).squeeze(2).squeeze(2)
        g2 = self.avgpool(x2).squeeze(2).squeeze(2)
        g3 = self.avgpool(x3).squeeze(2).squeeze(2)
        g4 = self.avgpool(x4).squeeze(2).squeeze(2)

        g2 = self.HGCNlinear1(g2)
        g3 = self.HGCNlinear2(g3)
        g4 = self.HGCNlinear3(g4)
        x = torch.cat((g1, g2, g3, g4), dim=1)
        return x

    def hyperG(self, A, knn, l1, sim, W):
        H = np.concatenate((A, knn, l1, sim), axis=1)
        # the degree of the node
        DV = np.sum(H, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        HT = H.T
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        DV2_H = torch.as_tensor(DV2_H).cuda().float()
        invDE_HT_DV2 = torch.as_tensor(invDE_HT_DV2).cuda().float()

        w = torch.diag(W)
        G = torch.mm(w, invDE_HT_DV2)
        G = torch.mm(DV2_H, G)
        return G

    def l1_representation(self, X, n_neighbors, gamma=1):
        n_nodes = X.shape[0]
        n_edges = n_nodes
        m_dist = pairwise_distances(X)
        m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors + 1]

        edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors + 1)).reshape(-1)
        node_idx = []
        values = []

        for i_edge in range(n_edges):
            neighbors = m_neighbors[i_edge].tolist()
            if i_edge in neighbors:
                neighbors.remove(i_edge)
            else:
                neighbors = neighbors[:-1]

            P = X[neighbors, :]  # k neighbor
            v = X[i_edge, :]  # ceneroid sample

            # cvxpy
            x = cp.Variable(P.shape[0], nonneg=True)
            objective = cp.Minimize(cp.norm((P.T @ x).T - v, 2) + gamma * cp.norm(x, 1))
            prob = cp.Problem(objective)
            try:
                prob.solve()
            except SolverError:
                prob.solve(solver='SCS', verbose=False)

            node_idx.extend([i_edge] + neighbors)
            values.extend([1.] + x.value.tolist())

        node_idx = np.array(node_idx)
        values = np.array(values)
        l1 = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges)).toarray()
        return l1

    def KNN(self, X, n_neighbors, is_prob=True):
        n_nodes = X.shape[0]
        n_edges = n_nodes

        m_dist = pairwise_distances(X)

        # top n_neighbors+1
        m_neighbors = np.argpartition(m_dist, kth=n_neighbors + 1, axis=1)
        m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

        m_neighbors = m_neighbors[:, :n_neighbors + 1]
        m_neighbors_val = m_neighbors_val[:, :n_neighbors + 1]

        # check
        for i in range(n_nodes):
            if not np.any(m_neighbors[i, :] == i):
                m_neighbors[i, -1] = i
                m_neighbors_val[i, -1] = 0.

        node_idx = m_neighbors.reshape(-1)
        edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors + 1)).reshape(-1)

        if not is_prob:
            values = np.ones(node_idx.shape[0])
        else:
            avg_dist = np.mean(m_dist)
            m_neighbors_val = m_neighbors_val.reshape(-1)
            values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

        knn = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges)).toarray()
        return knn

    def similarity(self, X, n_neighbors):
        n_nodes = X.shape[0]
        n_edges = n_nodes
        sim = np.zeros((n_nodes, n_edges))

        for i in range(n_nodes):
            dist = []
            for j in range(n_edges):
                s = X[i].dot(X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                dist.append(s)
            m_neighbors = sorted(dist, reverse=True)[0:n_neighbors + 1]
            for n in m_neighbors:
                ind = dist.index(n)
                sim[i][ind] = 1.0
        return sim


    def forward(self, img, A, pc):
        batch_size = img.size(0)
        img_len = img.size(1)
        x_feature = []
        for i in range(img_len):
            x = self.feature_exact(img[:, i, :, :, :].squeeze(1))
            x_feature.append(x.unsqueeze(1))

        x = torch.cat(x_feature, dim=1)
        X = x.cpu().detach().numpy()
        H = []
        n_neighbors = 4
        for j in range(batch_size):
            knn = self.KNN(X[j, :, :], n_neighbors)
            l1 = self.l1_representation(X[j, :, :], n_neighbors)
            sim = self.similarity(X[j, :, :], n_neighbors)
            G = self.hyperG(A[j, :, :], knn, l1, sim, self.HGCN_W)
            H.append(torch.as_tensor(G).unsqueeze(0))

        H = torch.cat(H, dim=0)
        gc5 = self.HGCN(x, H)
        out = gc5.view(batch_size, -1)
        img_score = torch.mean(out, dim=1).unsqueeze(1)

        pc_score = self.pcnet(pc)

        s = torch.cat((img_score, pc_score), dim=1)
        final_score = self.HGCN_linear6(s)
        return final_score


def get_model(pretrained=False, **kwargs):
    # get DHCN model
    model = DHCN(Bottleneck, [3, 4, 23, 3], **kwargs)
    # load pre-trained weight
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = torch.load('pre_weight/resnet101-5d3b4d8f.pth')
        pre_train_model = {k: v for k, v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
        for name, para in model.named_parameters():
            if "HGCN" not in name and "pcnet" not in name:
                para.requires_grad_(False)
    return model

