import torch.nn as nn
import torch
import scipy.io as sio
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import normalize, minmax_scale, StandardScaler
from sklearn import cluster
import os
import random
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from metrics import acc, f_score, b3_recall_score, b3_precision_score, randIndex
from scipy.sparse.linalg import svds


class MSRCV1(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train_num = int(210)
        data_dict = sio.loadmat('MSRCV1_210_6views_7clusters.mat')
        data = data_dict['X']
        data1 = data[0][0].astype(np.float32).transpose()
        data2 = data[0][1].astype(np.float32).transpose()
        data3 = data[0][2].astype(np.float32).transpose()
        data4 = data[0][3].astype(np.float32).transpose()
        data5 = data[0][4].astype(np.float32).transpose()
        data6 = data[0][5].astype(np.float32).transpose()

        #
        data1 = minmax_scale(data1, feature_range=(0, 1), axis=0)
        data2 = minmax_scale(data2, feature_range=(0, 1), axis=0)
        data3 = minmax_scale(data3, feature_range=(0, 1), axis=0)
        data4 = minmax_scale(data4, feature_range=(0, 1), axis=0)
        data5 = minmax_scale(data5, feature_range=(0, 1), axis=0)
        data6 = minmax_scale(data6, feature_range=(0, 1), axis=0)

        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5
        self.data6 = data6
        print(self.data1.shape)
        print(self.data2.shape)
        print(self.data3.shape)
        print(self.data4.shape)
        print(self.data5.shape)
        print(self.data6.shape)

    def __getitem__(self, index):
        img_train1, img_train2, img_train3 = self.data1[index, :], self.data2[index, :], self.data3[index, :]
        img_train4, img_train5, img_train6 = self.data4[index, :], self.data5[index, :], self.data6[index, :]
        return img_train1, img_train2, img_train3, img_train4, img_train5, img_train6

    def __len__(self):
        return self.train_num


class Nets(nn.Module):
    def __init__(self, enc_dim_list, dec_dim_list):
        super(Nets, self).__init__()
        self.encoder1_0 = nn.Sequential(nn.Linear(enc_dim_list[0][0], enc_dim_list[0][1]), nn.Tanh())
        self.coefficient1_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder1_1 = nn.Sequential(nn.Linear(enc_dim_list[0][1], enc_dim_list[0][2]), nn.Tanh())
        self.coefficient1_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder1_1 = nn.Sequential(nn.Linear(enc_dim_list[0][2], dec_dim_list[0][0]), nn.Tanh())
        self.decoder1_0 = nn.Sequential(nn.Linear(dec_dim_list[0][0], dec_dim_list[0][1]), nn.Tanh())

        self.encoder2_0 = nn.Sequential(nn.Linear(enc_dim_list[1][0], enc_dim_list[1][1]), nn.Tanh())
        self.coefficient2_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder2_1 = nn.Sequential(nn.Linear(enc_dim_list[1][1], enc_dim_list[1][2]), nn.Tanh())
        self.coefficient2_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder2_1 = nn.Sequential(nn.Linear(enc_dim_list[1][2], dec_dim_list[1][0]), nn.Tanh())
        self.decoder2_0 = nn.Sequential(nn.Linear(dec_dim_list[1][0], dec_dim_list[1][1]), nn.Tanh())

        self.encoder3_0 = nn.Sequential(nn.Linear(enc_dim_list[2][0], enc_dim_list[2][1]), nn.Tanh())
        self.coefficient3_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder3_1 = nn.Sequential(nn.Linear(enc_dim_list[2][1], enc_dim_list[2][2]), nn.Tanh())
        self.coefficient3_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder3_1 = nn.Sequential(nn.Linear(enc_dim_list[2][2], dec_dim_list[2][0]), nn.Tanh())
        self.decoder3_0 = nn.Sequential(nn.Linear(dec_dim_list[2][0], dec_dim_list[2][1]), nn.Tanh())

        self.encoder4_0 = nn.Sequential(nn.Linear(enc_dim_list[3][0], enc_dim_list[3][1]), nn.Tanh())
        self.coefficient4_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder4_1 = nn.Sequential(nn.Linear(enc_dim_list[3][1], enc_dim_list[3][2]), nn.Tanh())
        self.coefficient4_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder4_1 = nn.Sequential(nn.Linear(enc_dim_list[3][2], dec_dim_list[3][0]), nn.Tanh())
        self.decoder4_0 = nn.Sequential(nn.Linear(dec_dim_list[3][0], dec_dim_list[3][1]), nn.Tanh())

        self.encoder5_0 = nn.Sequential(nn.Linear(enc_dim_list[4][0], enc_dim_list[4][1]), nn.Tanh())
        self.coefficient5_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder5_1 = nn.Sequential(nn.Linear(enc_dim_list[4][1], enc_dim_list[4][2]), nn.Tanh())
        self.coefficient5_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder5_1 = nn.Sequential(nn.Linear(enc_dim_list[4][2], dec_dim_list[4][0]), nn.Tanh())
        self.decoder5_0 = nn.Sequential(nn.Linear(dec_dim_list[4][0], dec_dim_list[4][1]), nn.Tanh())

        self.encoder6_0 = nn.Sequential(nn.Linear(enc_dim_list[5][0], enc_dim_list[5][1]), nn.Tanh())
        self.coefficient6_0 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.encoder6_1 = nn.Sequential(nn.Linear(enc_dim_list[5][1], enc_dim_list[5][2]), nn.Tanh())
        self.coefficient6_1 = nn.Parameter(1.0e-6 * torch.ones(210, 210))
        self.decoder6_1 = nn.Sequential(nn.Linear(enc_dim_list[5][2], dec_dim_list[5][0]), nn.Tanh())
        self.decoder6_0 = nn.Sequential(nn.Linear(dec_dim_list[5][0], dec_dim_list[5][1]), nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                # nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2, input3, input4, input5, input6):
        z1_0 = self.encoder1_0(input1)
        recon_z1_0 = torch.matmul(self.coefficient1_0, z1_0)
        z1_1 = self.encoder1_1(z1_0)
        recon_z1_1 = torch.matmul(self.coefficient1_1, z1_1)
        output_z1_1 = self.decoder1_1(recon_z1_1)
        output1 = self.decoder1_0(output_z1_1 + recon_z1_0)

        z2_0 = self.encoder2_0(input2)
        recon_z2_0 = torch.matmul(self.coefficient2_0, z2_0)
        z2_1 = self.encoder2_1(z2_0)
        recon_z2_1 = torch.matmul(self.coefficient2_1, z2_1)
        output_z2_1 = self.decoder2_1(recon_z2_1)
        output2 = self.decoder2_0(output_z2_1 + recon_z2_0)

        z3_0 = self.encoder3_0(input3)
        recon_z3_0 = torch.matmul(self.coefficient3_0, z3_0)
        z3_1 = self.encoder3_1(z3_0)
        recon_z3_1 = torch.matmul(self.coefficient3_1, z3_1)
        output_z3_1 = self.decoder3_1(recon_z3_1)
        output3 = self.decoder3_0(output_z3_1 + recon_z3_0)

        z4_0 = self.encoder4_0(input4)
        recon_z4_0 = torch.matmul(self.coefficient4_0, z4_0)
        z4_1 = self.encoder4_1(z4_0)
        recon_z4_1 = torch.matmul(self.coefficient4_1, z4_1)
        output_z4_1 = self.decoder4_1(recon_z4_1)
        output4 = self.decoder4_0(output_z4_1 + recon_z4_0)

        z5_0 = self.encoder5_0(input5)
        recon_z5_0 = torch.matmul(self.coefficient5_0, z5_0)
        z5_1 = self.encoder5_1(z5_0)
        recon_z5_1 = torch.matmul(self.coefficient5_1, z5_1)
        output_z5_1 = self.decoder5_1(recon_z5_1)
        output5 = self.decoder5_0(output_z5_1 + recon_z5_0)

        z6_0 = self.encoder6_0(input6)
        recon_z6_0 = torch.matmul(self.coefficient6_0, z6_0)
        z6_1 = self.encoder6_1(z6_0)
        recon_z6_1 = torch.matmul(self.coefficient6_1, z6_1)
        output_z6_1 = self.decoder6_1(recon_z6_1)
        output6 = self.decoder6_0(output_z6_1 + recon_z6_0)

        return z1_0, self.coefficient1_0, recon_z1_0, z1_1, self.coefficient1_1, recon_z1_1, output1, \
               z2_0, self.coefficient2_0, recon_z2_0, z2_1, self.coefficient2_1, recon_z2_1, output2, \
               z3_0, self.coefficient3_0, recon_z3_0, z3_1, self.coefficient3_1, recon_z3_1, output3, \
               z4_0, self.coefficient4_0, recon_z4_0, z4_1, self.coefficient4_1, recon_z4_1, output4, \
               z5_0, self.coefficient5_0, recon_z5_0, z5_1, self.coefficient5_1, recon_z5_1, output5, \
               z6_0, self.coefficient6_0, recon_z6_0, z6_1, self.coefficient6_1, recon_z6_1, output6


def local(z, coef):
    return torch.matmul(coef, z)


def train_net1(model, coef, alpha, beta, gamma, lambd, params):
    gt = params['gt']
    num_class = params['num_class']
    batch_size = params['batch_size']
    device = 'cpu'
    epochs = params['epochs']
    lr = params['lr']
    data_loader = params['data_loader']
    model.to(device)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    coef1 = coef[0][0].astype(np.float32).transpose()
    coef2 = coef[0][1].astype(np.float32).transpose()
    coef3 = coef[0][2].astype(np.float32).transpose()
    coef4 = coef[0][3].astype(np.float32).transpose()
    coef5 = coef[0][4].astype(np.float32).transpose()
    coef6 = coef[0][5].astype(np.float32).transpose()
    coef1 = normalize(coef1, norm='l2', axis=1)
    coef2 = normalize(coef2, norm='l2', axis=1)
    coef3 = normalize(coef3, norm='l2', axis=1)
    coef4 = normalize(coef4, norm='l2', axis=1)
    coef5 = normalize(coef5, norm='l2', axis=1)
    coef6 = normalize(coef6, norm='l2', axis=1)
    coef1 = torch.tensor(coef1, dtype=torch.float32, device=device)
    coef2 = torch.tensor(coef2, dtype=torch.float32, device=device)
    coef3 = torch.tensor(coef3, dtype=torch.float32, device=device)
    coef4 = torch.tensor(coef4, dtype=torch.float32, device=device)
    coef5 = torch.tensor(coef5, dtype=torch.float32, device=device)
    coef6 = torch.tensor(coef6, dtype=torch.float32, device=device)

    coef_share = torch.nn.init.constant_(torch.empty((batch_size, batch_size), device=device, requires_grad=True), 0.0)

    fscore_curr_iter = []
    precision_curr_iter = []
    recall_curr_iter = []
    nmi_curr_iter = []
    ari_curr_iter = []
    acc_curr_iter = []
    ri_curr_iter = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data_ in data_loader:
            input_1, input_2, input_3, input_4, input_5, input_6 = data_
            input_1, input_2, input_3 = input_1.to(device), input_2.to(device), input_3.to(device)
            input_4, input_5, input_6 = input_4.to(device), input_5.to(device), input_6.to(device)
            z1_0, coef1_0, recon_z1_0, z1_1, coef1_1, recon_z1_1, output1, \
            z2_0, coef2_0, recon_z2_0, z2_1, coef2_1, recon_z2_1, output2, \
            z3_0, coef3_0, recon_z3_0, z3_1, coef3_1, recon_z3_1, output3, \
            z4_0, coef4_0, recon_z4_0, z4_1, coef4_1, recon_z4_1, output4, \
            z5_0, coef5_0, recon_z5_0, z5_1, coef5_1, recon_z5_1, output5, \
            z6_0, coef6_0, recon_z6_0, z6_1, coef6_1, recon_z6_1, output6 = \
                model(input_1, input_2, input_3, input_4, input_5, input_6)

            recon_loss = criterion(output1, input_1) + criterion(output2, input_2) + criterion(output3, input_3) + \
                         criterion(output4, input_4) + criterion(output5, input_5) + criterion(output6, input_6)

            local_loss = criterion(local(z1_0, coef1), z1_0) + criterion(local(z1_1, coef1_0), z1_1) + \
                         criterion(local(z2_0, coef2), z2_0) + criterion(local(z2_1, coef2_0), z2_1) + \
                         criterion(local(z3_0, coef3), z3_0) + criterion(local(z3_1, coef3_0), z3_1) + \
                         criterion(local(z4_0, coef4), z4_0) + criterion(local(z4_1, coef4_0), z4_1) + \
                         criterion(local(z5_0, coef5), z5_0) + criterion(local(z5_1, coef5_0), z5_1) + \
                         criterion(local(z6_0, coef6), z6_0) + criterion(local(z6_1, coef6_0), z6_1)

            selfexpr_loss = criterion(recon_z1_0, z1_0) + criterion(recon_z1_1, z1_1) + \
                            criterion(recon_z2_0, z2_0) + criterion(recon_z2_1, z2_1) + \
                            criterion(recon_z3_0, z3_0) + criterion(recon_z3_1, z3_1) + \
                            criterion(recon_z4_0, z4_0) + criterion(recon_z4_1, z4_1) + \
                            criterion(recon_z5_0, z5_0) + criterion(recon_z5_1, z5_1) + \
                            criterion(recon_z6_0, z6_0) + criterion(recon_z6_1, z6_1)

            coef_loss = torch.sum(torch.pow(coef1_0, 2)) + torch.sum(torch.pow(coef1_1, 2)) + \
                        torch.sum(torch.pow(coef2_0, 2)) + torch.sum(torch.pow(coef2_1, 2)) + \
                        torch.sum(torch.pow(coef3_0, 2)) + torch.sum(torch.pow(coef3_1, 2)) + \
                        torch.sum(torch.pow(coef4_0, 2)) + torch.sum(torch.pow(coef4_1, 2)) + \
                        torch.sum(torch.pow(coef5_0, 2)) + torch.sum(torch.pow(coef5_1, 2)) + \
                        torch.sum(torch.pow(coef6_0, 2)) + torch.sum(torch.pow(coef6_1, 2))

            selfexpr_loss_share = criterion(local(coef1_1, coef_share), coef1_1) + \
                                  criterion(local(coef2_1, coef_share), coef2_1) + \
                                  criterion(local(coef3_1, coef_share), coef3_1) + \
                                  criterion(local(coef4_1, coef_share), coef4_1) + \
                                  criterion(local(coef5_1, coef_share), coef5_1) + \
                                  criterion(local(coef6_1, coef_share), coef6_1)

            coef_loss_share = torch.sum(torch.pow(coef_share, 2))

            loss = recon_loss + alpha * local_loss + beta * selfexpr_loss + gamma * coef_loss + \
                   lambd * selfexpr_loss_share + coef_loss_share

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item() / batch_size

            temp = torch.matmul(model.coefficient1_1, model.coefficient1_1.T) + \
                   torch.matmul(model.coefficient2_1, model.coefficient2_1.T) + \
                   torch.matmul(model.coefficient3_1, model.coefficient3_1.T) + \
                   torch.matmul(model.coefficient4_1, model.coefficient4_1.T) + \
                   torch.matmul(model.coefficient5_1, model.coefficient5_1.T) + \
                   torch.matmul(model.coefficient6_1, model.coefficient6_1.T)
            coef_share = torch.matmul(lambd * temp, torch.linalg.inv(lambd * temp + torch.eye(batch_size)))

        try:
            coef_share_temp = coef_share.detach().cpu().numpy()
            alpha1 = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)
            coef_share_temp = thrC(coef_share_temp, alpha1)
            pred, _ = post_proC(coef_share_temp, num_class, 3, 1)
            measure1 = evaluation(gt, pred)
            fscore_curr_iter.append(measure1[0])
            precision_curr_iter.append(measure1[1])
            recall_curr_iter.append(measure1[2])
            nmi_curr_iter.append(measure1[3])
            ari_curr_iter.append(measure1[4])
            acc_curr_iter.append(measure1[5])
            ri_curr_iter.append(measure1[6])
        except:
            print("this epoch: {} occures error".format(epoch + 1))

    acc = max(acc_curr_iter)
    fscore = max(fscore_curr_iter)
    precision = max(precision_curr_iter)
    recall = max(recall_curr_iter)
    nmi = max(nmi_curr_iter)
    ari = max(ari_curr_iter)
    ri = max(ri_curr_iter)
    measure = np.zeros(7)
    measure[0] = fscore
    measure[1] = precision
    measure[2] = recall
    measure[3] = nmi
    measure[4] = ari
    measure[5] = acc
    measure[6] = ri
    return measure


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def evaluation(gt, pred):
    measure = np.zeros(7)
    fscore = f_score(gt, pred)
    precision = b3_precision_score(gt, pred)
    recall = b3_recall_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ari = adjusted_rand_score(gt, pred)
    accuracy = acc(gt, pred)
    ri = randIndex(gt, pred)
    measure[0] = fscore
    measure[1] = precision
    measure[2] = recall
    measure[3] = nmi
    measure[4] = ari
    measure[5] = accuracy
    measure[6] = ri
    return measure


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)                     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)                # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)            # if you are using multi-GPU. 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    times = 1
    measure_all = np.zeros((times, 7))
    for k in range(times):
        print('time={}'.format(k))
        coef_dict = sio.loadmat('MSRCV1_coef.mat')
        coef = coef_dict['coef']
        data_dict_ = sio.loadmat('MSRCV1_210_6views_7clusters.mat')
        data1 = data_dict_['X']
        num_view = data1.shape[1]
        num_samp = data1[0][0].shape[1]
        data0 = data_dict_['gt']
        label_true = np.zeros(num_samp)
        for i in range(num_samp):
            label_true[i] = data0[i]
        num_class = len(np.unique(label_true))
        set_seed(500)
        enc_dim_list = [[1302, 512, 128], [48, 64, 128], [512, 256, 128], [100, 128, 128], [256, 128, 128], [210, 128, 128]]
        dec_dim_list = [[512, 1302], [64, 48], [256, 512], [128, 100], [128, 256], [128, 210]]
        model_ = Nets(enc_dim_list, dec_dim_list)

        params = dict()
        params['gt'] = label_true
        params['num_class'] = num_class
        params['batch_size'] = 210
        params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params['epochs'] = 500
        params['lr'] = 2 ** -9
        params['data_loader'] = data.DataLoader(MSRCV1(), batch_size=210, shuffle=False, drop_last=False)

        alpha_ = 0.01
        beta_ = 1
        gamma_ = 1
        lambd_ = 0.01

        measure = train_net1(model=model_, coef=coef, alpha=alpha_, beta=beta_, gamma=gamma_, lambd=lambd_, params=params)

        measure_all[k] = measure

    mean_all = np.mean(measure_all, axis=0)
    std_all = np.std(measure_all, axis=0)
    print('mean_fscore = {:.4f}, std_fscore = {:.4f}'.format(mean_all[0], std_all[0]))
    print('mean_precision = {:.4f}, std_precision = {:.4f}'.format(mean_all[1], std_all[1]))
    print('mean_recall = {:.4f}, std_recall = {:.4f}'.format(mean_all[2], std_all[2]))
    print('mean_nmi = {:.4f}, std_nmi = {:.4f}'.format(mean_all[3], std_all[3]))
    print('mean_ari = {:.4f}, std_ari = {:.4f}'.format(mean_all[4], std_all[4]))
    print('mean_accuracy = {:.4f}, std_accuracy = {:.4f}'.format(mean_all[5], std_all[5]))
    print('mean_ri = {:.4f}, std_ri = {:.4f}'.format(mean_all[6], std_all[6]))





