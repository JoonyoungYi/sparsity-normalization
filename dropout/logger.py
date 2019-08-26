from common.utils import append_text_results
from common.utils import init_file


class Logger:
    def __init__(self, args):
        file_name = 'seed{seed}_in{input_normalization}_n-hunit{n_hidden_unit}_n-hlayer{n_hidden_layer}_dropout-rate{dropout_rate}_dropout-mode{dropout_mode}_lr{learning_rate}_max-epoch{max_epoch}'.format(
            seed=args.seed,
            input_normalization=args.input_normalization,
            n_hidden_unit=args.n_hidden_unit,
            n_hidden_layer=args.n_hidden_layer,
            dropout_rate=args.dropout_rate,
            dropout_mode=args.dropout_mode,
            learning_rate=args.learning_rate,
            max_epoch=args.max_epoch, )

        init_file('dropout', args.dataset, 'epoch', file_name)
        init_file('dropout', args.dataset, 'test-drop', file_name)

        self.file_name = file_name
        self.dataset = args.dataset

    def log_epoch(self, epoch_idx, train_rmse, test_rmse, min_test_rmse):
        file_name = self.file_name
        dataset = self.dataset
        append_text_results('dropout', dataset, 'epoch', file_name, "\t".join(
            str(r) for r in [epoch_idx, train_rmse, test_rmse, min_test_rmse]))

    def log_test_drop(self, rate, test_y_pred_mean):
        file_name = self.file_name
        dataset = self.dataset
        append_text_results('dropout', dataset, 'test-drop', file_name,
                            "\t".join(
                                str(r) for r in [rate, test_y_pred_mean]))
