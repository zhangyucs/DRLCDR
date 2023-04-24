import torch
import numpy as np
import tqdm
import pandas as pd


def eval_mae(model, data_loader, stage):
    print('Evaluating MAE:')
    model.eval()
    targets, predicts = list(), list()
    loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    with torch.no_grad():
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            pred = model(X, stage)
            print(np.shape(pred))
            targets.extend(y.squeeze(1).tolist())
            predicts.extend(pred.tolist())
    targets = torch.tensor(targets).float()
    predicts = torch.tensor(predicts)

    return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k( r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def test_one_user(r, k):
    hr = 1 if sum(r) > 0 else 0
    ndcg = ndcg_at_k(r, k, 1)
    return {'HR': hr, 'NDCG': ndcg}


def batch_user(users, batch_size):
    for i in range(0, len(users), batch_size):
        yield users[i:i+batch_size]


def Test(testDict, negDict, topk, model):
    print('[TEST]')
    results = {'HR': 0, 'NDCG': 0}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        users = list(testDict.keys())
        ratings = []

        for batch_users in batch_user(users, 1):
            groundTrue = testDict[batch_users[0]]
            negList = negDict[batch_users[0]]
            us = torch.tensor(batch_users).long()
            user_embed = model.get_embed(us)
            rating = model.get_ranking(user_embed)
            rating_pos = rating[0, groundTrue].cpu().numpy().tolist()
            rating_neg = rating[0, negList].cpu().numpy().tolist()
            for i in rating_pos:
                rating_neg.append(i)
                rk = torch.tensor(rating_neg).float().to(device)
                _, rating_K = torch.topk(rk, k=topk)
                rating_neg = rating_neg[:-1]
                rt=[1 if j==999 else 0 for j in rating_K.cpu()]
                ratings.append(rt)

        pre_results = []
        for r in ratings:
            # print(type(r))
            pre_results.append(test_one_user(r, topk))
        ntest = len(pre_results)
        for result in pre_results:
            results['HR'] += result['HR']
            results['NDCG'] += result['NDCG']
        results['HR'] /= float(ntest)
        results['NDCG'] /= float(ntest)
        return results


def train(data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
    print('Training Epoch {}:'.format(epoch))
    model.train()
    for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        if mapping:
            src_emb, tgt_emb = model(X, stage)
            loss = criterion(src_emb, tgt_emb)
        else:
            pred = model(X, stage)
            loss = criterion(pred, y.squeeze().float())
        model.zero_grad()
        loss.backward()
        optimizer.step()


def Test1(dataset, Recmodel, epoch, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    savepath = '../result/'+world.config['s_dataset']+'_'+world.config['t_dataset']+'/'
    # if multicore == 1:
    #     pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hr': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        neg_test_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            neg_test_list= dataset.read_neg_file()
            neg_test_ndarray = np.array(neg_test_list)
            item_list = neg_test_ndarray[batch_users][:]
            item_list = item_list.tolist()
            for i in range(len(batch_users)):
                item_list[i].extend(groundTrue[i])
            item_list = torch.Tensor(item_list).long()
            item_list = item_list.to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu, item_list)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(torch.full((len(batch_users),1), 999, dtype=torch.int))
        assert total_batch == len(users_list)

        X = zip(rating_list, groundTrue_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, savepath+str(epoch)))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hr'] /= float(len(users))

    print(results)
    return results


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test_one_batch(X, savepath):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    # print(r)
    f = open(savepath, 'a+')
    for r_ in r:
        f.write(str(r_)+'\n')
    f.close()
    pre, recall, ndcg, hr, ndcg_K = [], [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        hr.append(utils.hit_at_k(r, k))

    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr':np.array(hr)}