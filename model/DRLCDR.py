import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.singleVBGE import singleVBGE
from model.Conditional_VBGE import Conditional_VBGE
from utils.loss import kld_gauss

class DRLCDR(nn.Module):
    def __init__(self, opt):
        super(DRLCDR, self).__init__()
        self.opt=opt
        self.source_specific_GNN = singleVBGE(opt)
        self.source_sp_GNN = singleVBGE(opt)
        self.target_specific_GNN = singleVBGE(opt)
        self.target_sp_GNN = singleVBGE(opt)
        self.Conditional_GNN =  Conditional_VBGE(opt)
        self.dropout = opt["dropout"]
        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.save_KLD = []

        # if opt['init'] == 'normal':
        #     nn.init.normal_(self.source_user_embedding.weight, mean=0, std=1)
        #     nn.init.normal_(self.target_user_embedding.weight, mean=0, std=1)
        #     nn.init.normal_(self.source_item_embedding.weight, mean=0, std=1)
        #     nn.init.normal_(self.target_item_embedding.weight, mean=0, std=1)
        #     nn.init.normal_(self.source_user_embedding_share.weight, mean=0, std=1)
        #     nn.init.normal_(self.target_user_embedding_share.weight, mean=0, std=1)
        # elif opt['init'] == 'xavier':
        #     nn.init.xavier_uniform_(self.source_user_embedding.weight, gain=1)
        #     nn.init.xavier_uniform_(self.target_user_embedding.weight, gain=1)
        #     nn.init.xavier_uniform_(self.source_item_embedding.weight, gain=1)
        #     nn.init.xavier_uniform_(self.target_item_embedding.weight, gain=1)
        #     nn.init.xavier_uniform_(self.source_user_embedding_share.weight, gain=1)
        #     nn.init.xavier_uniform_(self.target_user_embedding_share.weight, gain=1)

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        return output

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, self.opt["condi_weight"] * kld_loss

    def save_KL(self):
        return self.save_KLD

    def forward(self, source_UV, source_VU, target_UV, target_VU, epoch=00):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_user_share = self.source_user_embedding_share(self.source_user_index)
        target_user_share = self.target_user_embedding_share(self.target_user_index)
        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, target_UV, target_VU)
        source_user_mean, source_user_sigma = self.source_sp_GNN.forward_user_share(source_user, source_UV, source_VU)
        target_user_mean, target_user_sigma = self.target_sp_GNN.forward_user_share(target_user, target_UV, target_VU)
        condi_source_mean, condi_source_sigma, condi_target_mean, condi_target_sigma = self.Conditional_GNN(source_user_share, target_user_share, \
                                                                                                            source_UV, source_VU, target_UV, target_VU, \
                                                                                                            source_learn_specific_user, target_learn_specific_user)
        condi_source_user, condi_source_kld_loss = self.reparameters(condi_source_mean, condi_source_sigma)
        condi_target_user, condi_target_kld_loss = self.reparameters(condi_target_mean, condi_target_sigma)
        source_condi_kld = kld_gauss(condi_source_mean, condi_source_sigma, source_user_mean, source_user_sigma)
        target_condi_kld = kld_gauss(condi_target_mean, condi_target_sigma, target_user_mean, target_user_sigma)
        source_target_kld = -kld_gauss(condi_source_mean, condi_source_sigma, condi_target_mean, condi_target_sigma)
        self.save_KLD = [source_condi_kld.item(), target_condi_kld.item(), source_target_kld.item()]
        source_learn_user = condi_source_user + source_learn_specific_user
        target_learn_user = condi_target_user + target_learn_specific_user
        kld_loss = condi_source_kld_loss + condi_target_kld_loss
        condi_Intra_loss = self.opt["condi_non_weight"] * source_condi_kld + self.opt["condi_non_weight"] * target_condi_kld
        condi_Inter_loss = self.opt["condi_condi_weight"] * source_target_kld
        self.kld_loss = kld_loss + condi_Intra_loss - condi_Inter_loss
        return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item

    def wramup(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          target_UV, target_VU)
        self.kld_loss = 0
        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item
