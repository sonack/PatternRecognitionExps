#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
from tqdm import tqdm

def read_data(data_path, dtype=np.float32):
    return np.genfromtxt(data_path, delimiter=',', dtype=dtype)


opt = {
    'max_epoch': 300,
    'batch_size': 32,
    'lr': 1e-2,
    'disp_freq': 100,
    'lr_decay_epochs': 100,
    'dropout_p': 0.1,
    'ensemble_num': 6,
}

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opt = dotdict(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# pdb.set_trace()


class MNIST_Dataset(Dataset):
    def __init__(self, data_arr, label_arr=None):
        self.data = data_arr
        self.label = label_arr
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ele = {
            'data': self.data[idx]
        }
        if not (self.label is None):
            ele['label'] = self.label[idx]
        return ele


def normalize_data(train_data):
    mean_vec = np.mean(train_data, axis=0)
    std_vec = np.std(train_data, axis=0)
    return mean_vec, std_vec

def split_train_val(data_path, label_path, split_proportion = [0.8, 0.1, 0.1]):
    assert len(split_proportion) == 3
    assert abs(np.array(split_proportion).sum() - 1.0) < 1e-10, 'split_proportion must added be 1!'

    data = read_data(data_path, dtype=np.float32)
    label = read_data(label_path, dtype=np.int64)

    n = len(data)
    split_point1 = int(n * split_proportion[0])
    split_point2 = int(n * (1 - split_proportion[2]))
    indices = np.arange(n)
    np.random.shuffle(indices)


    train_data = np.array([data[i] for i in indices[:split_point1]])
    train_label = np.array([label[i] for i in indices[:split_point1]])

    val_data = np.array([data[i] for i in indices[split_point1:split_point2]])
    val_label = np.array([label[i] for i in indices[split_point1:split_point2]])

    test_data = np.array([data[i] for i in indices[split_point2:]])
    test_label = np.array([label[i] for i in indices[split_point2:]])

    return train_data, train_label, val_data, val_label, test_data, test_label

def get_model():
    modules = [
        nn.Dropout(p=opt.dropout_p),
        nn.Linear(84, 128 * 8),
        nn.BatchNorm1d(128 * 8),
        nn.ReLU(),
        # nn.SELU(),

        nn.Dropout(p=opt.dropout_p),
        nn.Linear(128 * 8, 64 * 4),
        nn.BatchNorm1d(64 * 4),
        nn.ReLU(),
        # nn.SELU(),
        nn.Dropout(p=opt.dropout_p),
        nn.Linear(64 * 4, 32 * 4),
        nn.BatchNorm1d(32 * 4),
        nn.ReLU(),
        # nn.SELU(),

        nn.Dropout(p=opt.dropout_p),
        nn.Linear(32 * 4, 32 * 2),
        nn.BatchNorm1d(32 * 2),
        nn.ReLU(),
        # nn.SELU(),

        nn.Dropout(p=opt.dropout_p),
        nn.Linear(32 * 2, 10)
    ]
    return nn.Sequential(*modules)


def test(model, test_dataloader, loss_crit):
    model.eval()
    avg_test_loss = 0
    correct_cnt = 0
    tot_cnt = 0
    for sample_batched in test_dataloader:
        # pdb.set_trace()
        data = sample_batched['data'].to(device)
        label = sample_batched['label'].to(device)
        predict = model(data)
        # pdb.set_trace()
        loss = loss_crit(predict, label)
        avg_test_loss += loss.item() / len(test_dataloader)
        _, predict_ids = predict.max(1)
        correct_cnt += predict_ids.eq(label).sum().item()
        tot_cnt += len(data)
    # pdb.set_trace()
    print ( 'Avg test loss = %.15f, acc = %.4f%%  [%d/%d].' % (avg_test_loss, correct_cnt / tot_cnt * 100, correct_cnt, tot_cnt) )
    return correct_cnt / tot_cnt * 100

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # pdb.set_trace()
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():
    avg_test_acc = 0
    for ensemble_id in range(opt.ensemble_num):
        train_data_path = './TrainSamples.csv'
        label_data_path = './TrainLabels.csv'

        # train_data_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验4 分类器设计/Group 4/TrainSamples.csv'
        # label_data_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验4 分类器设计/Group 4/TrainLabels.csv'
        train_data, train_label, val_data, val_label, test_data, test_label = split_train_val(train_data_path, label_data_path, split_proportion=[0.9, 0, 0.1])
        # pdb.set_trace()
        print ('Train ensemble model %d' % ensemble_id)
        print ('=' * 30)
        print ('Train :', train_data.shape)
        print ('Val :', val_data.shape)
        print ('Test :', test_data.shape)
        mean_vec, std_vec = normalize_data(train_data)
        train_dataset = MNIST_Dataset( (train_data - mean_vec) / std_vec, train_label)
        train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = 4)

        # val_dataset = MNIST_Dataset( (val_data - mean_vec) / std_vec, val_label)
        # val_dataloader = DataLoader(val_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = 4)

        test_dataset = MNIST_Dataset( (test_data - mean_vec) / std_vec, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = 4)

        val_dataset = test_dataset
        val_dataloader = test_dataloader

        model = get_model()
        model.to(device)
        model.apply(weights_init)
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
        loss_crit = nn.CrossEntropyLoss()

        best_test_acc = 0

        for epoch in range(opt.max_epoch):
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr * (0.1 ** (epoch // opt.lr_decay_epochs))
                # pdb.set_trace()
            print ()
            print ('-' * 30)
            print ('Current lr = %f' % (opt.lr * (0.1 ** (epoch // opt.lr_decay_epochs))))
            avg_loss = 0
            model.train()

            correct_cnt = 0
            tot_cnt = 0

            for idx, sample_batched in enumerate(train_dataloader):
                # pdb.set_trace()
                data = sample_batched['data'].to(device)
                label = sample_batched['label'].to(device)
                # pdb.set_trace()
                predict = model(data)

                _, predict_ids = predict.max(1)
                correct_cnt += predict_ids.eq(label).sum().item()
                tot_cnt += len(data)

                loss = loss_crit(predict, label)
                avg_loss += loss.item() / len(train_dataloader)
                if idx % opt.disp_freq == 0:
                    print ('[%3d/%3d] [%3d/%3d] train loss = %.15f' % (idx, len(train_dataloader), epoch, opt.max_epoch, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # pdb.set_trace()
                # model()
            print ( 'Epoch [%3d/%3d] avg loss = %.15f, acc = %.4f%%  [%d/%d]' % (epoch, opt.max_epoch, avg_loss, correct_cnt / tot_cnt * 100, correct_cnt, tot_cnt ) )
            # test
            with torch.no_grad():
                test_acc = test(model, val_dataloader, loss_crit)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    print ('Best Test Acc: %.4f%%' % best_test_acc)
                    save_dict = {
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'mean_vec': mean_vec,
                        'std_vec': std_vec
                    }
                    torch.save(save_dict, 'ckpts/ckpt_%02d.pth' % ensemble_id)
                    print ('save best model_%02d!' % ensemble_id)
        avg_test_acc += best_test_acc
        # print ('Final Test:')
        # with torch.no_grad():
        #     avg_test_acc += test(model, test_dataloader, loss_crit)
    
        # save_dict = {
        #     'model': model.state_dict(),
        #     'optim': optimizer.state_dict(),
        #     'mean_vec': mean_vec,
        #     'std_vec': std_vec
        # }
        # torch.save(save_dict, 'ckpts/ckpt_%02d.pth' % ensemble_id)
        # print ('save model_%02d!' % ensemble_id)
    print ('Avg Test Acc = %.4f%%.' % (avg_test_acc / opt.ensemble_num))
       


def eval(test_data_path):
    data = read_data(test_data_path, dtype=np.float32)
    eval_dataset = MNIST_Dataset(data)
    eval_dataloader = DataLoader(eval_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = 4)
     
    scores = np.empty((len(data), 10, opt.ensemble_num), dtype=np.float32)
    models = [get_model() for i in range(opt.ensemble_num)]
    for i_model, model in tqdm(enumerate(models)):
        ckpt = torch.load('ckpts/ckpt_%02d.pth' % i_model)
        mean_vec = torch.tensor(ckpt['mean_vec']).to(device)
        std_vec = torch.tensor(ckpt['std_vec']).to(device)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(eval_dataloader)):
                data = sample_batched['data'].to(device)
                scores[i_batch*opt.batch_size:(i_batch+1)*opt.batch_size, : , i_model] = model((data - mean_vec) / std_vec)

    predict = np.argmax(scores.sum(axis=-1), axis=1)
    with open('Result.csv', 'w') as f:
        for p in predict:
            f.write(str(p)+'\n')

if __name__ == '__main__':
    # train()
    # eval('./TrainSamples.csv')
    eval('./TestSamples.csv')

    # pdb.set_trace()