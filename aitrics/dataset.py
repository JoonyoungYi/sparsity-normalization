import pickle, os
import numpy as np
from sklearn.utils import class_weight

'''Requirements of init_data
args.data_path
args.data_path에 data가 있어야함
args.choose_dataset은 1~5
args.use_modified는 True/False
args.model은 mlp/lstm
args.normalize_input
'''

'''Output of init_data
X_train, y_train, X_test, y_test, class_weights, disease
'''

def init_data(args):

    # get dataset name
    diseases = os.listdir(args.data_path)
    disease = diseases[args.choose_dataset - 1]

    # whether to use modified dataset
    if args.use_modified:
        file_name = "/datasets_modified.pkl"
        sequence_length = 5
    else:
        file_name = "/datasets.pkl"
        sequence_length = 4

    # load dataset
    data_file = args.data_path + '/' + disease + file_name
    data = pickle.load(open(data_file, "rb"))

    # training data setting
    X_train = data["train_inputs"]
    X_train = X_train.reshape((-1, sequence_length*34))

    # [0,1] normalization
    max_feature = np.amax(X_train, 0)
    max_feature = max_feature + (max_feature < 1e-5) # zero to one
    X_train /= max_feature

    if args.normalize_input:
        train_num_records = data["train_num_records"]*34
        L0_norm = np.expand_dims(train_num_records, 1)
        X_train /= L0_norm

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (-1, sequence_length, 34))

    y_train = data["train_labels"]
    y_train = np.ravel(y_train)

    # validation data setting
    X_test = data["valid_inputs"]
    X_test = X_test.reshape((-1 ,sequence_length*34))

    # [0,1] normalization by using max_feature of training data
    X_test /= max_feature

    if args.normalize_input:
        valid_num_records = data["valid_num_records"]*34
        L0_norm = np.expand_dims(valid_num_records, 1)
        X_test /= L0_norm

    if args.model == 'lstm':
        X_test = np.reshape(X_test, (-1, sequence_length, 34))

    y_test = data["valid_labels"]
    y_test = np.ravel(y_test)

    # compute class weights for handling imbalance
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    class_weights = dict(zip([0, 1], class_weights))

    return X_train, y_train, X_test, y_test, class_weights, disease
