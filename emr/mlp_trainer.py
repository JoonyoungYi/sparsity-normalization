import argparse

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.utils as torch_utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from common.utils import set_torch_seed


def _precision_recall_auc(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    _auc = auc(recall, precision)
    return _auc


def _validate(model, criterion, _test_x, _test_y):
    # print(_test_x.shape)
    test_y_pred = model(_test_x)
    test_loss = criterion(test_y_pred, _test_y)

    test_y_prob = torch.sigmoid(test_y_pred)
    test_y_prob = test_y_prob.detach().cpu().numpy()
    _test_y = _test_y.detach().cpu().numpy().astype(np.int32)

    auc_func = roc_auc_score
    # auc_func = _precision_recall_auc
    test_auc = auc_func(_test_y, test_y_prob)
    # print(test_auc)
    # test_auc = auc_func(_test_y, test_y_prob, average="weighted")
    # print(test_auc)
    # print(auc_func(_test_y, test_y_prob, average="micro"))
    # print(auc_func(_test_y, test_y_prob, average="samples"))
    # print(auc_func(_test_y, test_y_prob, average=None))
    # input('!')

    # _test_y_predict = (test_y_prob >= 0.5).astype(np.int32)
    # print(recall_score(_test_y, _test_y_prob))
    # print(_test_y_prob)
    # print(_test_y)

    # input('!')
    # test_auc = np.sum(_test_y == _test_y_predict) / _test_y.shape[0]
    # input('!')

    # test_auc = f1_score(_test_y.astype(np.int32), (test_y_prob >= 0.5).astype(np.int32) )
    return test_auc, test_loss


def _test(model, _test_x, _test_y):
    test_y_pred = model(_test_x)
    test_y_prob = torch.sigmoid(test_y_pred)
    test_y_prob = test_y_prob.detach().cpu().numpy()
    _test_y = _test_y.detach().cpu().numpy().astype(np.int32)

    auc_func = _precision_recall_auc
    test_auprc = auc_func(_test_y, test_y_prob)
    # print('>> test AUPRC:', test_auprc)

    return test_auprc


def main():

    N_FEATURE = 36

    parser = argparse.ArgumentParser(description='mlp')
    args = parser.parse_args()
    args.n_batch = 500
    args.n_timestep = 288
    args.learning_rate = 1e-5
    args.max_epoch = 10000
    args.early_stop_epoch = 1000
    # args.use_class_weight = False
    args.use_class_weight = True
    args.max_grad_norm = None
    # args.max_grad_norm = 1e-2

    # args.mode = "ZI"  #  zero imputation
    args.mode = "SN"  # sparsity_normalization

    args.seed = 1

    set_torch_seed(args.seed)

    base_path = 'emr/datasets/physionet2012/processed/t{n_timestep}-{normalization}-dseed{dataset_seed}-'.format(
        normalization='minmax',
        # normalization='standardization',
        n_timestep=args.n_timestep,
        dataset_seed=1)

    train_x = np.load(base_path + 'train_x.npy')
    train_y = np.load(base_path + 'train_y.npy')[:, 0]
    n_train = train_x.shape[0]

    valid_x = np.load(base_path + 'valid_x.npy')
    valid_y = np.load(base_path + 'valid_y.npy')[:, 0]

    test_x = np.load(base_path + 'test_x.npy')
    test_y = np.load(base_path + 'test_y.npy')[:, 0]

    # test_x[np.random.random(test_x.shape[0]) < 0.5] = 0
    print(train_x.shape)
    print(valid_x.shape)
    print(test_x.shape)

    # input('!')

    # print(test_x.shape)
    # test_x[:, 12:, :] = 0
    # input('!!')

    # input('!')

    # def _normalize_input(train_x, valid_x, test_x):
    #     for k in range(N_FEATURE):
    #         _x = train_x[:, :, k]
    #         _c = np.sum(~np.isnan(_x), axis=1, keepdims=True)
    #         K_prime = np.mean(_c)
    #
    #         print(k, K_prime)
    #         # _x[np.isnan(_x)] = 0.0
    #         train_x[:, :, k] = train_x[:, :, k] / K_prime
    #         valid_x[:, :, k] = valid_x[:, :, k] / K_prime
    #         test_x[:, :, k] = test_x[:, :, k] / K_prime
    #         # input('!')
    #
    #     return train_x, valid_x, test_x

    # train_x, valid_x, test_x = _normalize_input(train_x, valid_x, test_x)

    if args.mode == "ZI":
        train_x[np.isnan(train_x)] = 0.0
        valid_x[np.isnan(valid_x)] = 0.0
        test_x[np.isnan(test_x)] = 0.0

    elif args.mode == "SN":
        K_prime = np.sum(~np.isnan(train_x)) / (
            train_x.shape[0] * train_x.shape[1])
        print('K_prime:', K_prime)

        K = np.sum(~np.isnan(train_x), axis=1, keepdims=True)
        K = np.mean(K, axis=0, keepdims=True) / 4
        # print(K.shape)
        # print(K)
        # input('!')

        def _normalize(train_x):
            train_c = np.sum(~np.isnan(train_x), axis=2, keepdims=True)
            train_c = np.maximum(train_c, 1.0)
            train_x[np.isnan(train_x)] = 0.0
            train_x = train_x * K / train_c
            return train_x

        train_x = _normalize(train_x)
        valid_x = _normalize(valid_x)
        test_x = _normalize(test_x)

    else:
        raise NotImplementedError()

    train_x = train_x.reshape([-1, args.n_timestep * N_FEATURE])
    valid_x = valid_x.reshape([-1, args.n_timestep * N_FEATURE])
    test_x = test_x.reshape([-1, args.n_timestep * N_FEATURE])

    print(np.mean(train_x))
    print(np.mean(valid_x))
    print(np.mean(test_x))

    assert np.sum(np.isnan(train_x)) == 0
    assert np.sum(np.isnan(test_x)) == 0

    # Here we define our model as a class
    class LSTM(torch.nn.Module):
        def __init__(self,
                     input_dim=N_FEATURE * args.n_timestep,
                     hidden_dim=50,
                     batch_size=args.n_batch,
                     output_dim=1,
                     num_layers=3):
            super(LSTM, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size
            self.num_layers = num_layers

            linears = [torch.nn.Linear(input_dim, hidden_dim)]
            for _ in range(num_layers - 1):
                linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            linears.append(torch.nn.Linear(hidden_dim, output_dim))
            self.linears = torch.nn.ModuleList(linears)

            # Define the LSTM layer
            # self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim,
            #                           self.num_layers)
            # self.lstm = torch.nn.RNN(
            #     self.input_dim,
            #     self.hidden_dim,
            #     num_layers=self.num_layers,
            #     bias=True,
            #     nonlinearity='relu',
            #     batch_first=False,
            #     dropout=0,
            #     bidirectional=False)

            # Define the output layer
            # self.linear = torch.nn.Linear(self.hidden_dim, output_dim)

        def forward(self, x):
            linears = self.linears

            for linear in linears[:-1]:
                x = linear(x)
                # x = F.relu(x)
                x = F.leaky_relu(x)
                # x = F.dropout(x, 0.5, training=self.training)
            y_pred = linears[-1](x).view(-1)

            # # print(x.shape)
            # # Forward pass through LSTM layer
            # # shape of lstm_out: [input_size, batch_size, hidden_dim]
            # # shape of self.hidden: (a, b), where a and b both
            # # have shape (num_layers, batch_size, hidden_dim).
            # lstm_out, _ = self.lstm(x)
            #
            # # Only take the output from the final timetep
            # # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
            # # tail pooling
            # # print(lstm_out.shape)
            # # out = lstm_out.mean(0)
            # out = lstm_out[-1]
            # # print(out.shape)
            # # print(lstm_out[-1].shape)
            # # input('!')
            # y_pred = self.linear(out)
            return y_pred.view(-1)

    # print(train_x.shape)
    # train_x = np.moveaxis(train_x, 0, 1)
    # print(train_x.shape)
    labels, counts = np.unique(train_y, return_counts=True)
    assert labels.shape[0] == 2
    assert counts[0] > counts[1]
    _pos_weight = counts[0] / counts[1]
    print('>> pos_weight: {:f}'.format(_pos_weight))

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(torch.FloatTensor).cuda()
    # train_w = torch.from_numpy(train_w).type(torch.FloatTensor).cuda()

    # valid_x = np.moveaxis(valid_x, 0, 1)
    valid_x = torch.from_numpy(valid_x).type(torch.FloatTensor).cuda()
    valid_y = torch.from_numpy(valid_y).type(torch.FloatTensor).cuda()

    # test_x = np.moveaxis(test_x, 0, 1)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).type(torch.FloatTensor).cuda()

    model = LSTM()  # .cuda()
    model.cuda()
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('>> Total params is {}'.format(total_params))
    # input('!')

    pos_weight = torch.from_numpy(np.array([_pos_weight])).type(
        torch.FloatTensor).cuda()
    if args.use_class_weight:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    test_criterion = torch.nn.BCEWithLogitsLoss()
    # test_criterion = criterion

    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-4)

    min_test_loss = float('inf')
    min_test_epoch_idx = -1
    min_test_epoch_test_auc = -1
    min_valid_loss = float('inf')
    min_valid_epoch_idx = -1
    min_valid_epoch_valid_auc = -1
    min_valid_epoch_test_auc = -1
    test_auprc = float('inf')
    max_test_auc = -1
    max_valid_auc = -1
    max_valid_epoch_idx = -1
    max_valid_epoch_test_auc = -1
    for epoch_idx in range(1, args.max_epoch + 1):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.train()
        for start_idx in range(0, n_train, args.n_batch):
            batch_x = train_x[start_idx:start_idx + args.n_batch, :]
            batch_y = train_y[start_idx:start_idx + args.n_batch]
            # print(batch_x)
            # print(batch_y)

            model.zero_grad()

            # Forward pass
            batch_y_pred = model(batch_x)
            loss = criterion(batch_y_pred, batch_y)

            optimiser.zero_grad()
            loss.backward()
            if args.max_grad_norm:
                torch_utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)
            optimiser.step()

        model.eval()
        valid_auc, valid_loss = _validate(model, test_criterion, valid_x,
                                          valid_y)
        test_auc, test_loss = _validate(model, test_criterion, test_x, test_y)

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            min_test_epoch_idx = epoch_idx
            min_test_epoch_test_auc = test_auc

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            min_valid_epoch_idx = epoch_idx
            min_valid_epoch_valid_auc = valid_auc
            min_valid_epoch_test_auc = test_auc

        max_test_auc = max(test_auc, max_test_auc)
        if max_valid_auc < valid_auc:
            max_valid_auc = valid_auc
            max_valid_epoch_test_auc = test_auc
            max_valid_epoch_idx = epoch_idx
            test_auprc = _test(model, test_x, test_y)

        if max_valid_epoch_idx + args.early_stop_epoch < epoch_idx:
            break

        print('')
        print(
            '%8d' % epoch_idx,
            '%.6f' % valid_loss,
            '%.6f' % test_loss,
            '%.6f' % valid_auc,
            '%.6f' % test_auc,
            '%.6f' % max_valid_auc,
            '%.6f' % max_test_auc,
            '%8d' % max_valid_epoch_idx,
            '%.6f' % max_valid_epoch_test_auc, )
        print(
            '%8d' % min_test_epoch_idx,
            '%.6f' % min_test_loss,
            '%.6f' % min_test_epoch_test_auc,
            '%8d' % min_valid_epoch_idx,
            '%.6f' % min_valid_loss,
            '%.6f' % min_valid_epoch_valid_auc,
            '%.6f' % min_valid_epoch_test_auc,
            '%.6f' % test_auprc, )


if __name__ == '__main__':
    main()
