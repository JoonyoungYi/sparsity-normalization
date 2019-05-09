import os

assert __name__ == '__main__'

for dataset in range(1, 6):
    # mlp
    os.system("python3 -m aitrics.trainer --model=mlp --epochs=200 --learning_rate=1e-4 --gpu_memory_fraction=0.01 --verbose=1 --normalize_input=False --choose_dataset={dataset} --seed=1".format(dataset=dataset))

    # lstm
    #os.system("python3 -m aitrics.trainer --model=lstm --epochs=100 --learning_rate=1e-3 --gpu_memory_fraction=0.1 --verbose=1 --normalize_input=False --choose_dataset={dataset} --seed=1".format(dataset=dataset))
