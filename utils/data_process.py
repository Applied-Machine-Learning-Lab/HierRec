import shutil
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import datetime
import copy
import torch
import datatable as dt
from collections import Counter

class ReadDataset(Dataset):
    def __init__(self, data_dir=None, name=None, train='train', shuffle=False):
        self.name = name
        if train not in ['train', 'val', 'test']:
            raise ValueError("parameter 'train' must be one of ['train', 'val', 'test']")
        else:
            self.train = train
        self.domain = None
        if self.name is None:
            self.field = None
            self.label = None
            return
        print("Loading {} data for {}...".format(self.train, self.name))
        if self.name=="aliccp":
            self.field = dt.fread(data_dir+'ali_ccp_'+self.train+'_20.csv',
                                  columns=lambda cols:[col.name not in ('purchase','D109_14','D110_14',
                       'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853') for col in cols])
            # self.field = dt.fread(data_dir + 'ali_ccp_' + self.train + '_20.csv',
            #                       columns=lambda cols: [col.name not in ('purchase', 'D109_14', 'D110_14',
            #                                                              'D127_14', 'D150_14', 'D508', 'D509', 'D702',
            #                                                              'D853') for col in cols], max_nrows=100000)
            self.label = self.field[:,"click"].to_numpy().astype(np.int_).squeeze()
            domain = self.field[:,"301"].to_numpy().astype(np.int_)
            del self.field[: , ["click", "301"]]
            self.field = np.hstack((domain, self.field.to_numpy().astype(np.int_)))
        elif self.name=="douban":
            self.field = dt.fread(data_dir + '/douban_data_split/{}/douban_music_'.format(self.train) + self.train + '.csv', header=True).to_numpy().astype(np.int_)
            self.field = np.vstack((self.field, dt.fread(data_dir + '/douban_data_split/{}/douban_book_'.format(self.train) + self.train + '.csv',  header=True).to_numpy().astype(np.int_)))
            self.field = np.vstack((self.field, dt.fread(data_dir + '/douban_data_split/{}/douban_movie_'.format(self.train) + self.train + '.csv',  header=True).to_numpy().astype(np.int_)))
            # self.field = dt.fread(
            #     data_dir + '/douban_data_split/{}/douban_music_'.format(self.train) + self.train + '.csv',
            #     header=True, max_nrows=30000).to_numpy().astype(np.int_)
            # self.field = np.vstack((self.field, dt.fread(
            #     data_dir + '/douban_data_split/{}/douban_book_'.format(self.train) + self.train + '.csv',
            #     header=True, max_nrows=30000).to_numpy().astype(np.int_)))
            # self.field = np.vstack((self.field, dt.fread(
            #     data_dir + '/douban_data_split/{}/douban_movie_'.format(self.train) + self.train + '.csv',
            #     header=True, max_nrows=30000).to_numpy().astype(np.int_)))
            self.label = self.field[:,3]
            self.field = self.field[:,:3]
        elif self.name=='kuairand':
            # Scenario Counter({0: 4431299, 2: 1341158, 1: 492851, 3: 218737, 6: 117358, 10: 17698, 4: 9718,
            # 7: 9433, 11: 7843, 5: 7230, 9: 2137, 14: 779, 13: 439, 12: 364, 8: 17})
            self.field = dt.fread(data_dir + 'kuairand.csv')
            # self.field = dt.fread(data_dir + 'kuairand.csv',max_nrows=100000)
            self.label = self.field[:, "is_click"].to_numpy().astype(np.int_).squeeze()
            domain = self.field[:, "tab"].to_numpy().astype(np.int_)
            del self.field[:, ["is_click", "tab", ]]
            self.field = np.hstack((domain, self.field.to_numpy().astype(np.int_)))

            select_domain=[0,1,2,3,6]
            mapping = dict(zip(select_domain, [i for i in range(len(select_domain))]))
            domain_mask = None
            for domain_id in select_domain:
                if domain_mask is None:
                    domain_mask = self.field[:,0]==domain_id
                else:
                    domain_mask += self.field[:,0]==domain_id
            self.field = self.field[domain_mask]
            self.label = self.label[domain_mask]

            # map selected to new id [0-end]
            k = np.array(list(mapping.keys()))
            v = np.array(list(mapping.values()))
            mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
            mapping_ar[k] = v
            self.field[:,0] = mapping_ar[self.field[:,0]]


        self.field = self.field-np.min(self.field, axis=0)
        print("Feature fields:", self.field.shape[1])
        print("Samples in each domain: ", Counter(self.field[:,0]))
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(self.field)
            np.random.set_state(state)
            np.random.shuffle(self.label)

        return
        raise ValueError('unknown dataset name: ' + name)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

    def field_dims(self):
        return np.max(self.field, axis=0) + 1


    def split(self, part=[0.8]):
        if self.field is not None and self.label is not None:
            len_set = len(self.label)
            part = np.cumsum(np.array([i * len_set for i in part]), dtype=int)
            part = np.append(part, len_set)
            # print("PARTITION: ", part)
            all_set = []
            start = 0
            for end in part:
                tmp_set = ReadDataset()
                tmp_set.field = self.field[start:end, :]
                tmp_set.label = self.label[start:end]
                all_set.append(tmp_set)
                start=end
            return all_set
        else:
            raise ValueError('no interaction in the dataset.')

    def sample(self, ratio):
        if 1.0 > ratio > 0.0:
            tmp_set = ReadDataset()
            tmp_set.field = self.field
            tmp_set.label = self.label
            state = np.random.get_state()
            np.random.shuffle(tmp_set.field)
            np.random.set_state(state)
            np.random.shuffle(tmp_set.label)
            sample_num = int(len(tmp_set.field) * ratio)
            tmp_set.field = tmp_set.field[:sample_num, :]
            tmp_set.label = tmp_set.label[:sample_num]
            return tmp_set

    def domain_dv(self):
        domain_set = []
        domain_ind = self.field[:,0]
        for i in range(self.field_dims()[0]):
            tmp_set = ReadDataset()
            di = np.argwhere(domain_ind == i).squeeze()
            tmp_set.field = self.field[di]
            tmp_set.label = self.label[di]
            tmp_set.domain = i
            domain_set.append(tmp_set)
        return domain_set


# dataset = ReadDataset("../data/", "aliccp")
# print(dataset.field[:,0])