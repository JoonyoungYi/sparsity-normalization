import os

assert __name__ == '__main__'

for seed in range(1, 6):
    # mlp
    #os.system("python3 -m aitrics.trainer --model=mlp --gpu_memory_fraction=0.06 --epochs_early_stop=5000 --seed={} --choose_dataset=1 --dropout=False --batch_norm_mlp=False --normalize_input=False".format(seed))

    # lstm
    #os.system("python3 -m aitrics.trainer --model=lstm --gpu_memory_fraction=0.28 --epochs_early_stop=5000 --seed={} --choose_dataset=1 --dropout=False --layer_norm_lstm=False --normalize_input=False".format(dataset))
