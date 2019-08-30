import re
import os
import glob

for file_path in glob.glob('.results/dropout/boston/test-drop/*.txt'):
    m = re.search('\.results\/dropout\/boston\/test-drop\/(?P<name>.*)\.txt',
                  file_path)
    name = m.group('name')
    assert name

    m = re.search(
        'seed1_instandardization_n-hunit(?P<n_hidden_unit>\d+)_n-hlayer(?P<n_hidden_layer>\d+)_dropout-rate(?P<dropout_rate>.+)_dropout-mode(?P<dropout_mode>(default|sparsity-normalization))_lr(?P<learning_rate>.+)_max-epoch(?P<max_epoch>.+)_act(?P<activation_fn>.+)_n-batch0',
        name)
    if m is None:
        assert name.startswith("2*50")
        continue

    newname = '2*50_act-{activation_fn}_adam({learning_rate})_dropout-rate{dropout_rate}_dropout-mode{dropout_mode}_max-epoch{max_epoch}_n-batch0_instandardization_seed1'.format(
        n_hidden_unit=m.group('n_hidden_unit'),
        n_hidden_layer=m.group('n_hidden_layer'),
        dropout_rate=m.group('dropout_rate'),
        dropout_mode=m.group('dropout_mode'),
        learning_rate=m.group('learning_rate'),
        max_epoch=m.group('max_epoch'),
        activation_fn=m.group('activation_fn'))
    # print(newname)

    os.rename(file_path,
              '.results/dropout/boston/test-drop/{}.txt'.format(newname))
