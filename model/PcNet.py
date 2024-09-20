import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
from torch.nn import Parameter
from model.pointnet2 import Pointnet2
import cvxpy as cp
from cvxpy.error import SolverError

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


class PCNet(nn.Module):
    def __init__(self):
        super(PCNet, self).__init__()
        img_len = 6
        self.W = Parameter(torch.ones(img_len * 3))
        self.HGCN = HGCNNet(img_len=img_len)

        self.relu = nn.ReLU()
        self.pc_backbone = Pointnet2()
        self.pc_inplanes = 1024

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

    def hyperG(self, knn, l1, sim, W):
        H = np.concatenate((knn, l1, sim), axis=1)
        # H = knn
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

    def forward(self, pc):
        # extract features from patches
        pc_size = pc.shape
        pc = pc.view(-1, pc_size[2], pc_size[3])
        pc = self.pc_backbone(pc)

        # average the patch features
        pc = pc.view(pc_size[0], pc_size[1], self.pc_inplanes)

        X = pc.cpu().detach().numpy()
        H = []
        n_neighbors = 1
        for j in range(pc_size[0]):
            knn = self.KNN(X[j, :, :], n_neighbors)
            l1 = self.l1_representation(X[j, :, :], n_neighbors)
            sim = self.similarity(X[j, :, :], n_neighbors)

            G = self.hyperG(knn, l1, sim, self.W)
            H.append(torch.as_tensor(G).unsqueeze(0))

        H = torch.cat(H, dim=0)

        gc5 = self.HGCN(pc, H)

        out = gc5.view(pc_size[0], -1)
        score = torch.mean(out, dim=1).unsqueeze(1)
        return score