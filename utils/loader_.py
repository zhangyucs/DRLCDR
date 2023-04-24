"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs
import pandas as pd

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation

        # ************* source data *****************
        source_train_data = "../dataset/" + filename + "/train.txt"
        source_test_data = "../dataset/" + filename + "/test.txt"
        source_save_user_path = "../dataset/" + filename
        source_test_neg_data = "../dataset/" + filename + "/test_neg.log"
        # self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_test_data, self.source_user, self.source_item = self.read_data(source_train_data, source_test_data, source_test_neg_data)
        self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_test_data, self.source_user, self.source_item = self.read_data(source_train_data, source_test_data, source_save_user_path)

        opt["source_user_num"] = len(self.source_user)
        opt["source_item_num"] = len(self.source_item)
        # ************* target data *****************
        filename = filename.split("_")
        filename = filename[1] + "_" + filename[0]
        target_train_data = "../dataset/" + filename + "/train.txt"
        target_test_data = "../dataset/" + filename + "/test.txt"
        target_save_user_path = "../dataset/" + filename
        target_test_neg_data = "../dataset/" + filename + "/test_neg.log"
        # self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_test_data, self.target_user, self.target_item = self.read_data(target_train_data, target_test_data, target_test_neg_data)
        self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_test_data, self.target_user, self.target_item = self.read_data(target_train_data, target_test_data, target_save_user_path)

        opt["target_user_num"] = len(self.target_user)
        opt["target_item_num"] = len(self.target_item)

        # print(np.shape(self.target_ma_set), np.shape(self.source_ma_set))

        # opt["rate"] = self.rate()

        assert opt["source_user_num"] == opt["target_user_num"]
        if evaluation == -1:
            data = self.preprocess()
        else :
            data = self.preprocess_for_predict()
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_data(self, train_file, test_file, test_neg_file=None):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = {}
            item = {}
            ma = {}
            ma_list = {}
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    user[line[0]] = len(user)
                if item.get(line[1], "zxczxc") is "zxczxc":
                    item[line[1]] = len(item)
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                train_data.append([line[0],line[1]])
                if line[0] not in ma:
                    ma[line[0]] = set()
                    ma_list[line[0]] = []
                ma[line[0]].add(line[1])
                ma_list[line[0]].append(line[1])
        negd = {}
        with open(test_neg_file, 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split('\n')
                arr = arr[0].split('\t')
                # arr = line.split(',')
                u = eval(arr[0].split(',')[0].split('(')[1])
                arr=list(map(int,arr[1:]))
                negItem = arr[0:]
                negd[u] = negItem
                line = fd.readline()
        with codecs.open(test_file,"r",encoding="utf-8") as infile:
            test_data=[]
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])

                if user.get(line[0], "zxczxc") is "zxczxc":
                    continue
                if item.get(line[1], "zxczxc") is "zxczxc":
                    # print(line[1])
                    continue
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                ret = [line[1]]
                ret.extend(negd[line[0]])
                test_data.append([line[0],ret])

        return ma, ma_list, train_data, test_data, user, item

    def preprocess_for_predict(self):
        processed=[]
        if self.eval == 1:
            for d in self.source_test_data:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        else :
            for d in self.target_test_data:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        return processed
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.source_train_data:
            d = [d[1], d[0]]
            processed.append(d + [-1])
        for d in self.target_train_data:
            processed.append([-1] + d)
        return processed

    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1 :
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            user = []
            for b in batch:
                if b[0] == -1:
                    source_pos_tmp.append(self.find_pos(self.source_ma_list, b[1]))
                    target_pos_tmp.append(b[2])
                else:
                    source_pos_tmp.append(b[0])
                    target_pos_tmp.append(self.find_pos(self.target_ma_list, b[1]))
                source_neg_tmp.append(self.find_neg(self.source_ma_set, b[1], "source_item_num"))
                target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                user.append(b[1])
            return (torch.LongTensor(user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)