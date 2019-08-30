from common.utils import append_text_results
from common.utils import init_file


def get_file_name(n_hidden_layer, n_hidden_unit, activation_fn, optimizer,
                  learning_rate, dropout_rate, dropout_mode, max_epoch,
                  n_batch, input_normalization, seed):
    return ('{n_hidden_layer}*{n_hidden_unit}'
            '_act-{activation_fn}'
            '_{optimizer}({learning_rate})'
            '_dropout-rate{dropout_rate}'
            '_dropout-mode{dropout_mode}'
            '_max-epoch{max_epoch}'
            '_n-batch{n_batch}'
            '_in{input_normalization}'
            '_seed{seed}').format(
                seed=seed,
                input_normalization=input_normalization,
                n_hidden_unit=n_hidden_unit,
                n_hidden_layer=n_hidden_layer,
                dropout_rate=dropout_rate,
                dropout_mode=dropout_mode,
                learning_rate=learning_rate,
                activation_fn=activation_fn,
                n_batch=n_batch if n_batch else 0,
                optimizer=optimizer,
                max_epoch=max_epoch, )


class Logger:
    def __init__(self, args):
        file_name = get_file_name(
            seed=args.seed,
            input_normalization=args.input_normalization,
            n_hidden_unit=args.n_hidden_unit,
            n_hidden_layer=args.n_hidden_layer,
            dropout_rate=args.dropout_rate,
            dropout_mode=args.dropout_mode,
            learning_rate=args.learning_rate,
            activation_fn=args.activation_fn,
            n_batch=args.n_batch,
            optimizer=args.optimizer,
            max_epoch=args.max_epoch)

        init_file('dropout', args.dataset, 'epoch', file_name)
        init_file('dropout', args.dataset, 'test-drop', file_name)

        self.file_name = file_name
        self.dataset = args.dataset

    def log_epoch(self, epoch_idx, batch_rmse, train_rmse, valid_rmse, test_rmse):
        file_name = self.file_name
        dataset = self.dataset
        append_text_results('dropout', dataset, 'epoch', file_name, "\t".join(
            str(r)
            for r in
            [epoch_idx, batch_rmse, train_rmse, valid_rmse, test_rmse]))

    def log_test_drop(self, rate, test_y_pred_mean):
        file_name = self.file_name
        dataset = self.dataset
        append_text_results('dropout', dataset, 'test-drop', file_name,
                            "\t".join(
                                str(r) for r in [rate, test_y_pred_mean]))
