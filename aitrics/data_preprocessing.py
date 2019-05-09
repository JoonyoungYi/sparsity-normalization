from tqdm import tqdm
import pickle, os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='AITRICS_PREPROCESSING')

parser.add_argument(
    '--data_path',
    type=str,
    default='aitrics/data',
    help='e.g. [data_path]/cardiovascular1')

args = parser.parse_args()

# get dataset names
dirs = os.listdir(args.data_path)

# for each disease dataset, change the format and save as datasets_modified.pkl
for disease in dirs:
    print('Data: {}'.format(disease))

    # get data file name
    data_file = args.data_path + '/' + disease + "/datasets.pkl"

    ''' for a rainy day
    # load data
    data = pickle.load(open(data_file, "rb"))

    # data manipulation

    # missing value calculation

    # make possibly missing mask (possibly missing: 0, otherwise: 1)
    possibly_missing_features = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 24, 25, 26, 27, 28]
    possibly_missing_mask = np.ones(34)

    for idx in possibly_missing_features:
        possibly_missing_mask[idx] = 0

    # get data
    train_inputs = data['train_inputs']
    valid_inputs = data['valid_inputs']

    train_num_records = data['train_num_records']
    valid_num_records = data['valid_num_records']

    # init missing number array
    train_num_missing = np.array([], dtype=int)
    valid_num_missing = np.array([], dtype=int)

    # calculate number of missing value and fill the array
    for i in range(train_inputs.shape[0]):
        for j in range(train_num_records[i]):
            train_inputs[i][j] += possibly_missing_mask
        train_num_missing = np.append(train_num_missing, np.count_nonzero(train_inputs[i] < 1e-5))

    for i in range(valid_inputs.shape[0]):
        for j in range(valid_num_records[i]):
            valid_inputs[i][j] += possibly_missing_mask
        valid_num_missing = np.append(valid_num_missing, np.count_nonzero(valid_inputs[i] < 1e-5))
    '''
    # change input sequence per patient e.g. [2011 2013, 0, 0] -> [0, 0, 2011, 0, 2013]

    # load data again
    data = pickle.load(open(data_file, "rb"))

    # get data
    train_inputs = data['train_inputs']
    valid_inputs = data['valid_inputs']

    train_years = data['train_years']
    valid_years = data['valid_years']

    # prepare manipulation
    modified_train_inputs = np.zeros((train_inputs.shape[0], 5, 34))
    modified_valid_inputs = np.zeros((valid_inputs.shape[0], 5, 34))
    mold = np.zeros((5, 34))

    # manipulate input sequence
    for i in tqdm(range(train_inputs.shape[0])):
        for idx, j in enumerate(train_years[i] - 2009):
            mold[j] = train_inputs[i][idx]
        modified_train_inputs[i] = mold
        mold *= 0

    for i in tqdm(range(valid_inputs.shape[0])):
        for idx, j in enumerate(valid_years[i] - 2009):
            mold[j] = valid_inputs[i][idx]
        modified_valid_inputs[i] = mold
        mold *= 0

    # prepare modified data
    data['train_inputs'] = modified_train_inputs
    data['valid_inputs'] = modified_valid_inputs

    ''' for a rainy day
    data['train_num_missing'] = train_num_missing
    data['valid_num_missing'] = valid_num_missing
    '''

    # save data
    modified_data_file = args.data_path + disease + "/datasets_modified.pkl"
    f = open(modified_data_file ,"wb")

    pickle.dump(data, f)
    f.close()
