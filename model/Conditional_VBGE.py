import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
# from model.GCN import VGAE
# from torch.autograd import Variable
# from torch.distributions.kl import kl_divergence
# from torch.distributions import Normal
from utils.loss import kld_gauss

class Conditional_VBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(Conditional_VBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]
        self.condi_source_user = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.condi_target_user = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.isCondi_norm = opt["isCondi_norm"]
        self.isConditional = opt['isConditional']


    # def get_augmented_features(self, x, conditional_feature, concat=1):
    #     X_list = []
    #     # cvae_features = torch.tensor(features, dtype=torch.float32).cuda()
    #     # for _ in range(concat):
    #     #     z = torch.randn([cvae_features.size(0), self.latent_size]).cuda()
    #     #     augmented_features = cvae_model.inference(z, cvae_features)
    #     #     augmented_features = self.feature_tensor_normalize(augmented_features).detach()
    #     #     if self.gpu:
    #     #         X_list.append(augmented_features.cuda())
    #     #     else:
    #     #         X_list.append(augmented_features)
    #     # print(np.shape(cvae_features), np.shape(z), np.shape(augmented_features))
    #     # print(cvae_features.size(0), cvae_model.latent_size)

    #     features = x
    #     cvae_features = torch.tensor(features, dtype=torch.float32).cuda()
    #     for _ in range(concat):
    #         z = torch.randn([cvae_features.size(0), self.latent_size]).cuda()
    #         augmented_features = cvae_model.inference(z, cvae_features)
    #         augmented_features = self.feature_tensor_normalize(augmented_features).detach()
    #         if self.gpu:
    #             X_list.append(augmented_features.cuda())
    #         else:
    #             X_list.append(augmented_features)
    #     print(np.shape(cvae_features), np.shape(z), np.shape(augmented_features))
    #     print(cvae_features.size(0), self.latent_size)

    #     return X_list

    def feature_tensor_normalize(self, feature):
        rowsum = torch.div(1.0, torch.sum(feature, dim=1))
        rowsum[torch.isinf(rowsum)] = 0.
        feature = torch.mm(torch.diag(rowsum), feature)
        return feature

    def conditional(self, s_feature, s_condi_feature, t_feature, t_condi_feature):
        condi_source = torch.cat((s_feature, t_condi_feature), dim=1)
        condi_target = torch.cat((t_feature, s_condi_feature), dim=1)
        if self.isCondi_norm:
            user_source = self.feature_tensor_normalize(F.relu(self.condi_source_user(condi_source)))
            user_target = self.feature_tensor_normalize(F.relu(self.condi_target_user(condi_target)))
        else:
            user_source = F.relu(self.condi_source_user(condi_source))
            user_target = F.relu(self.condi_target_user(condi_target))
        return user_source, user_target

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj, condi_source_ufea, condi_target_ufea):
        if self.isConditional == False:
            learn_user_source = source_ufea
            learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            if self.isConditional:
                learn_user_source, learn_user_target = self.conditional(source_ufea, condi_source_ufea, target_ufea, condi_target_ufea)
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
        if self.isConditional:
            learn_user_source, learn_user_target = self.conditional(source_ufea, condi_source_ufea, target_ufea, condi_target_ufea)
        source_User_mean, source_User_logstd, target_User_mean, target_User_logstd = self.encoder[-1](learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
        return source_User_mean, source_User_logstd, target_User_mean, target_User_logstd

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], 
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)
        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)
        source_User = torch.cat((source_User_ho , source_ufea), dim=1)
        source_User = self.source_user_union(source_User)
        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)
        return F.relu(source_User), F.relu(target_User)

class LastLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.source_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.source_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)
        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)
        source_User_mean = torch.cat((source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)
        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)
        target_User_mean = torch.cat((target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)
        target_User_logstd = torch.cat((target_User_ho_logstd, target_ufea),dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)
        return source_User_mean, source_User_logstd, target_User_mean, target_User_logstd


