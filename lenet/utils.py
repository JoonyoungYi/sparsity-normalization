import argparse
import os
import random

import numpy as np
import torch


def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')


def write_result_to_file(args, result_dict, file_name):
    args_dict = vars(args)

    args_keys_list = sorted([key for key in args_dict.keys()])
    args_list = [str(args_dict[key]) for key in args_keys_list]
    args_line = '\t'.join(args_list)

    result_keys_list = sorted([key for key in result_dict.keys()])
    result_list = [str(result_dict[key]) for key in result_keys_list]
    result_line = '\t'.join(result_list)

    line = '\t'.join([args_line, result_line]) + '\n'

    if not os.path.exists(file_name):
        args_keys = '\t'.join(args_keys_list)
        result_keys = '\t'.join(result_keys_list)
        keys = '\t'.join([args_keys, result_keys]) + '\n'
        with open(file_name, 'w') as res:
            res.write(keys)

    with open(file_name, 'a') as res:
        res.write(line)
