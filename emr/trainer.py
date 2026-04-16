import argparse

import numpy as np
import torch
import torch.nn.utils as torch_utils
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from common.utils import set_torch_seed


def _precision_recall_auc(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return auc(recall, precision)


def _validate(model, criterion, _test_x, _test_y):
    with torch.no_grad():
        test_y_pred = model(_test_x)
        test_loss = criterion(test_y_pred, _test_y)
        test_y_prob = torch.sigmoid(test_y_pred).cpu().numpy()
    _test_y_np = _test_y.cpu().numpy().astype(np.int32)
    test_auc = roc_auc_score(_test_y_np, test_y_prob)
    return test_auc, test_loss


def _test(model, _test_x, _test_y):
    with torch.no_grad():
        test_y_pred = model(_test_x)
        test_y_prob = torch.sigmoid(test_y_pred).cpu().numpy()
    _test_y_np = _test_y.cpu().numpy().astype(np.int32)
    return _precision_recall_auc(_test_y_np, test_y_prob)


def main():
    set_torch_seed(1)
    N_FEATURE = 36

    parser = argparse.ArgumentParser(description='Physionet2012 mortality prediction')
    parser.add_argument('--mode', type=str, default='SN',
                        choices=['ZI', 'MI', 'FI', 'SN'],
                        help='ZI: zero imputation, MI: mean imputation, '
                             'FI: forward imputation, SN: sparsity normalization')
    parser.add_argument('--sn_imputation', type=str, default='none',
                        choices=['none', 'MI', 'FI'])
    args = parser.parse_args()

    args.n_batch = 500
    args.n_timestep = 48
    args.learning_rate = 1e-5
    args.max_epoch = 10000
    args.early_stop_epoch = 1000
    args.use_class_weight = True
    args.max_grad_norm = None

    base_path = 'emr/datasets/physionet2012/processed/t{n_timestep}-standardization-dseed1-'.format(
        n_timestep=args.n_timestep)

    train_x = np.load(base_path + 'train_x.npy')
    train_y = np.load(base_path + 'train_y.npy')[:, 0]
    n_train = train_x.shape[0]

    valid_x = np.load(base_path + 'valid_x.npy')
    valid_y = np.load(base_path + 'valid_y.npy')[:, 0]

    test_x = np.load(base_path + 'test_x.npy')
    test_y = np.load(base_path + 'test_y.npy')[:, 0]

    print(train_x.shape, valid_x.shape, test_x.shape)

    if args.mode == "ZI":
        train_x[np.isnan(train_x)] = 0.0
        valid_x[np.isnan(valid_x)] = 0.0
        test_x[np.isnan(test_x)] = 0.0

    elif args.mode == "MI":
        K_prime = np.sum(~np.isnan(train_x)) / (train_x.shape[0] * train_x.shape[1])
        feature_mean = np.zeros(N_FEATURE)
        for j in range(N_FEATURE):
            vals = train_x[:, :, j][~np.isnan(train_x[:, :, j])]
            feature_mean[j] = np.mean(vals)
            train_x[:, :, j][np.isnan(train_x[:, :, j])] = feature_mean[j]
            valid_x[:, :, j][np.isnan(valid_x[:, :, j])] = feature_mean[j]
            test_x[:, :, j][np.isnan(test_x[:, :, j])] = feature_mean[j]
        train_x *= K_prime
        valid_x *= K_prime
        test_x *= K_prime

    elif args.mode == "FI":
        feature_mean = np.array([
            np.mean(train_x[:, :, j][~np.isnan(train_x[:, :, j])])
            for j in range(N_FEATURE)
        ])

        for split_x in [train_x, valid_x, test_x]:
            for i in range(split_x.shape[0]):
                _x = split_x[i]
                __x = feature_mean.copy()
                for t in range(_x.shape[0]):
                    _x[t] = np.where(np.isnan(_x[t]), __x, _x[t])
                    __x = _x[t]

    elif args.mode == "SN":
        K_prime = np.sum(~np.isnan(train_x)) / (train_x.shape[0] * train_x.shape[1])

        if args.sn_imputation == "MI":
            feature_mean = np.array([
                np.mean(train_x[:, :, j][~np.isnan(train_x[:, :, j])])
                for j in range(N_FEATURE)
            ])
            for split_x in [train_x, valid_x, test_x]:
                for i in range(split_x.shape[0]):
                    _x = split_x[i]
                    for t in range(_x.shape[0]):
                        if np.sum(~np.isnan(_x[t])) == 0:
                            _x[t] = feature_mean

        elif args.sn_imputation == "FI":
            def _forward_impute(data):
                for i in range(data.shape[0]):
                    _x, __x = data[i], data[i][0].copy()
                    for t in range(_x.shape[0]):
                        if np.sum(~np.isnan(_x[t])) == 0:
                            _x[t] = __x
                        __x = _x[t]
                return data

            def _backward_impute(data):
                for i in range(data.shape[0]):
                    _x = data[i]
                    __x = _x[-1].copy()
                    for t in range(_x.shape[0] - 1, -1, -1):
                        if np.sum(~np.isnan(_x[t])) == 0:
                            _x[t] = __x
                        __x = _x[t]
                return data

            for split_x in [train_x, valid_x, test_x]:
                _forward_impute(split_x)
                _backward_impute(split_x)

        def _normalize(data):
            c = np.sum(~np.isnan(data), axis=2, keepdims=True).astype(np.float32)
            c = np.maximum(c, 1.0)
            data[np.isnan(data)] = 0.0
            return data * K_prime / c

        train_x = _normalize(train_x)
        valid_x = _normalize(valid_x)
        test_x = _normalize(test_x)

    else:
        raise NotImplementedError()

    assert np.sum(np.isnan(train_x)) == 0
    assert np.sum(np.isnan(test_x)) == 0

    class LSTM(torch.nn.Module):
        def __init__(self, input_dim=N_FEATURE, hidden_dim=50, output_dim=1):
            super(LSTM, self).__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=1)
            self.linear = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.linear(lstm_out[-1]).view(-1)

    labels, counts = np.unique(train_y, return_counts=True)
    assert labels.shape[0] == 2
    _pos_weight = counts[0] / counts[1]
    print('>> pos_weight: {:f}'.format(_pos_weight))

    train_x = torch.from_numpy(np.moveaxis(train_x, 0, 1)).float().cuda()
    train_y = torch.from_numpy(train_y).float().cuda()
    valid_x = torch.from_numpy(np.moveaxis(valid_x, 0, 1)).float().cuda()
    valid_y = torch.from_numpy(valid_y).float().cuda()
    test_x = torch.from_numpy(np.moveaxis(test_x, 0, 1)).float().cuda()
    test_y = torch.from_numpy(test_y).float().cuda()

    model = LSTM().cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('>> Total params: {}'.format(total_params))

    pos_weight = torch.tensor([_pos_weight]).float().cuda()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight if args.use_class_weight else None)
    test_criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)

    max_valid_auc = -1
    max_valid_epoch_idx = -1
    max_valid_epoch_test_auc = -1
    test_auprc = -1

    for epoch_idx in range(1, args.max_epoch + 1):
        model.train()
        for start_idx in range(0, n_train, args.n_batch):
            batch_x = train_x[:, start_idx:start_idx + args.n_batch, :]
            batch_y = train_y[start_idx:start_idx + args.n_batch]
            model.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if args.max_grad_norm:
                torch_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimiser.step()

        model.eval()
        valid_auc, valid_loss = _validate(model, test_criterion, valid_x, valid_y)
        test_auc, _ = _validate(model, test_criterion, test_x, test_y)

        if max_valid_auc < valid_auc:
            max_valid_auc = valid_auc
            max_valid_epoch_test_auc = test_auc
            max_valid_epoch_idx = epoch_idx
            test_auprc = _test(model, test_x, test_y)

        if max_valid_epoch_idx + args.early_stop_epoch < epoch_idx:
            break

        print('%8d' % epoch_idx,
              '%.6f' % valid_auc, '%.6f' % test_auc,
              '%.6f' % max_valid_auc, '%.6f' % max_valid_epoch_test_auc,
              '%.6f' % test_auprc)

    print('\nFinal - AUROC: {:.6f}, AUPRC: {:.6f}'.format(
        max_valid_epoch_test_auc, test_auprc))


if __name__ == '__main__':
    main()
