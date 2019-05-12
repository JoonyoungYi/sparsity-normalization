python3 -m cfnade.trainer --gpu_memory_fraction=0.22 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0 --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True

python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.1 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0 --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True

python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.1 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0 --parameter_sharing=True --use_ordinal_cost=True --normalize_input=False
