from common.utils import make_folder_if_not_exist
from common.utils import append_text_results
from common.utils import init_file


def _init_folder_and_files(dataset):
    make_folder_if_not_exist('.results/')
    make_folder_if_not_exist('.results/lenet/')
    make_folder_if_not_exist('.results/lenet/{}/'.format(dataset))
    make_folder_if_not_exist('.results/lenet/{}/epoch/'.format(dataset))
    make_folder_if_not_exist('.results/lenet/{}/model/'.format(dataset))


def _get_model_name(args):
    return '{}_{:.1f}_{}'.format(args.use_batch_normalization,
                                 args.dropout_rate, args.dropout_mode).lower()

def get_epoch_file_name(args):
    return _get_model_name(args) + '.tsv'

def get_model_file_name(args):
    return _get_model_name(args) + '.pt.latest'


class Logger:
    def __init__(self, args):
        dataset = 'cifar10'
        epoch_file_name = get_epoch_file_name(args)
        model_file_name = get_model_file_name(args)
        init_file('lenet', dataset, 'epoch', epoch_file_name)
        _init_folder_and_files(dataset)

        self.epoch_file_name = epoch_file_name
        self.dataset = dataset

    def log_epoch(self, epoch_idx, result):
        dataset = self.dataset
        epoch_file_name = self.epoch_file_name

        (train_loss, train_acc), (valid_loss, valid_acc), (test_loss,
                                                           test_acc) = result
        append_text_results('lenet', dataset, 'epoch', epoch_file_name,
                            "\t".join(
                                str(r)
                                for r in [
                                    epoch_idx, train_loss, train_acc,
                                    valid_loss, valid_acc, test_loss, test_acc
                                ]))
