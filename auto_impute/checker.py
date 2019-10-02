import argparse

import numpy as np

parser = argparse.ArgumentParser(description='auto_impute')
args = parser.parse_args()
# args.dataset = 'jurkat'  # 0.17487025714038504
# args.dataset = 'preimplantation' # 0.45752693717545595
# args.dataset = 'blakeley'  # 0.17685655323810073
# args.dataset = 'kolodziejczyk' # 0.10898919561010287
# args.dataset = 'quake' # 0.33256329610897856
# args.dataset = 'usoskin' # 0.48689693973640025
# args.dataset = 'zeisel'  # 0.3566374502560199
args.data = args.dataset + '.csv'

# reading dataset
try:
    extn = args.data.split('.')[1]
    if (extn == 'mat'):
        print("[!data read] Reading from data/" + args.data)
        processed_count_matrix = scipy.io.mmread("auto_impute/data/" +
                                                 args.data)
        processed_count_matrix = processed_count_matrix.toarray()
        processed_count_matrix = np.array(processed_count_matrix)
    else:
        print("[!data read] Reading from data/" + args.data)
        with open("auto_impute/data/" + args.data) as f:
            ncols = len(f.readline().split(','))
        processed_count_matrix = np.loadtxt(
            open("auto_impute/data/" + args.data, "rb"),
            delimiter=",",
            skiprows=1,
            usecols=range(1, ncols + 1))
except:
    from traceback import format_exc
    print(format_exc())
    print(
        "[!data read] Please make sure that your processed dataset is in data/ in .mat or .csv format and you have entered the filename as data parameter. e.g. blakeley.csv"
    )
    exit()

print(processed_count_matrix)
c = np.sum(processed_count_matrix > 1e-20, axis=0)
print(np.mean(c))
print(np.std(c))
print(np.std(c) / np.mean(c))
# print(np.std(c) * np.std(c) / np.mean(c))
