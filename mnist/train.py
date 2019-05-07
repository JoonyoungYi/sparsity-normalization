import os

assert __name__ == '__main__'

for n_hidden_unit in range(1, 100):
    os.system("python3 -m mnist.trainer --n_hidden_unit={n_hidden_unit}".
              format(n_hidden_unit=n_hidden_unit))
