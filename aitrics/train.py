import os

assert __name__ == '__main__'

for dataset in range(1, 6):
    # mlp
    #os.system("python3 -m aitrics.trainer --model=mlp --gpu_memory_fraction=0.06 --choose_dataset={} --seed=1 --dropout=False --normalize_input=False".format(dataset))

    # lstm
    #os.system("python3 -m aitrics.trainer --model=lstm --gpu_memory_fraction=0.24 --choose_dataset={} --seed=1 --dropout=False --normalize_input=False".format(dataset))
