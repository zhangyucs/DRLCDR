import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import numpy as np

def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def kld_gauss(mu_1, logsigma_1, mu_2, logsigma_2):
    """Using std to compute KLD"""
    sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
    sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
    # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
    # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
    q_target = Normal(mu_1, sigma_1)
    q_context = Normal(mu_2, sigma_2)
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
    return kl

def jsd_gauss(mu_1, logsigma_1, mu_2, logsigma_2):
    """Using std to compute JSD"""
    sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
    sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
    # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
    # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
    q_target = Normal(mu_1, sigma_1)
    q_context = Normal(mu_2, sigma_2)
    m_q_target = F.softmax(q_target.sample(), dim=0)
    m_q_context = F.softmax(q_context.sample(), dim=0)
    m = 0.5*(m_q_target+m_q_context)
    js_loss = 0.0
    js_loss += F.kl_div(F.log_softmax(m_q_context, dim=0), m, reduction='batchmean')
    js_loss += F.kl_div(F.log_softmax(m_q_target, dim=0), m, reduction='batchmean')
    # q_kl = kl_divergence(q_target, m).mean(dim=0).sum()
    # p_kl = kl_divergence(q_context, m).mean(dim=0).sum()
    # return np.mean(0.5*q_kl + 0.5*p_kl)
    return (0.5*js_loss)

def kld(z_1, z_2):
    kl = F.kl_div(F.softmax(z_1, dim=0), F.softmax(z_2, dim=0), reduction='batchmean')
    return kl