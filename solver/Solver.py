import torch
import numpy as np
from model.DHCN import get_model
import math
from tqdm import tqdm
from scipy import stats
from data.dataloader import *
from scipy.optimize import curve_fit


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted


class Model_Solver(object):
    """training and testing"""
    def __init__(self, config, path):
        self.config = config
        self.epochs = config.epochs
        self.mn = config.model_name
        self.lr = config.lr
        self.bs = config.batch_size
        self.split = config.split
        self.path = path
        self.weight_decay = config.weight_decay
        self.l1_loss = torch.nn.L1Loss().cuda()

        # Loading Model
        self.model = get_model(pretrained=True).cuda()
        self.model_path = "save weight path/" + config.dataset
        self.model.train(True)
        self.resume_epoch = 0

        self.img_params = []
        self.pc_params = []
        for name, para in self.model.named_parameters():
            if "HGCN" in name:
                self.img_params.append(para)
            elif "pcnet" in name:
                self.pc_params.append(para)

        paras = [{'params': self.model.parameters(), 'lr': self.lr}]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = DataLoader(self.config, self.config.dataset, self.path, self.split, self.config.image_size,
                                  self.bs, istrain=True)
        test_loader = DataLoader(self.config, self.config.dataset, self.path, self.split, self.config.image_size,
                                 istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        no_save_epoch = 0
        best_srcc = 0.0
        best_plcc = 0.0

        if self.config.resume:
            self.resume_epoch = ""
            self.model.load_state_dict(torch.load("path"))

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores_all = []
            gt_scores_all = []
            for img, label, A, pc in tqdm(self.train_data):
                img = torch.as_tensor(img.cuda())
                pc = torch.as_tensor(pc.cuda())
                label = torch.as_tensor(label.cuda())

                self.optimizer.zero_grad()

                pred = self.model(img, A, pc)
                loss = self.l1_loss(pred.squeeze(1), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                pred_scores_all = pred_scores_all + pred.cpu().tolist()
                gt_scores_all = gt_scores_all + label.cpu().tolist()

            train_srcc, _ = stats.spearmanr(pred_scores_all, gt_scores_all)
            train_plcc, _ = stats.pearsonr(np.squeeze(pred_scores_all), np.squeeze(gt_scores_all))

            test_srcc, test_plcc, test_krcc, test_RMSE = self.test(self.test_data)
            if test_srcc > best_srcc:
                no_save_epoch = 0
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model.state_dict(), self.model_path +
                           "/{}_epoch{}_lr{}_bs{}_split{}_srcc{:.4f}.pth".
                           format(self.mn, t+1+self.resume_epoch, self.lr, self.bs, self.split, best_srcc))
            else:
                no_save_epoch += 1

            print('EPOCH:%d\tLOSS:%4.3f\tTrain_SRCC:%4.4f\tTrain_PLCC:%4.4f\tTest_SRCC:%4.4f\tTest_PLCC:%4.4f\tTest_KRCC:%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, test_srcc, test_plcc, test_krcc))

            if no_save_epoch == 20:
                print("20 epoch did not improve performance, stop train...")
                break

            # lr drop
            lr_down = pow(0.5, (t // 10))
            paras = [{'params': self.model.parameters(), 'lr': self.lr*lr_down}]  # 初始2e-3
            self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        return best_srcc, best_plcc

    def test(self, data, model_path=None):
        """Testing"""
        if model_path is not None:
            model_dict = self.model.state_dict()
            pre_train_model = torch.load(model_path)
            pre_train_model = {k: v for k, v in pre_train_model.items() if k in model_dict}
            model_dict.update(pre_train_model)
            self.model.load_state_dict(model_dict)
            print('load the pretrained model, done！')
            self.model.eval()

        self.model.train(False)
        pred_scores = []
        gt_scores = []
        for img, label, A, pc in data:
            img = torch.as_tensor(img.cuda())
            pc = torch.as_tensor(pc.cuda())

            # predict score
            pred = self.model(img, A, pc)

            if self.config.dataset == "wpc":
                pred_scores.append(float(pred.item() * 10))
                gt_scores = gt_scores + (label * 10).tolist()
            else:
                pred_scores.append(float(pred.item()))
                gt_scores = gt_scores + (label).tolist()

        # non-linear mapping
        _, _, pred_scores = logistic_5_fitting_no_constraint(pred_scores, gt_scores)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(np.squeeze(pred_scores), np.squeeze(gt_scores))
        test_krcc, _ = stats.kendalltau(pred_scores, gt_scores)
        test_RMSE = self.get_RMSE(gt_scores, pred_scores)

        self.model.train(True)
        return test_srcc, test_plcc, test_krcc, test_RMSE

    def get_RMSE(self, records_real, records_predict):
        """
        Root Mean Squard Error
        """
        if len(records_real) == len(records_predict):
            return math.sqrt(sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real))
        else:
            return None

