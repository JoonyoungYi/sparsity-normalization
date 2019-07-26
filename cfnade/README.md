# python3 -m cfnade.trainer --gpu_memory_fraction=0.22 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0 --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True
# python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.1 --learning_rate=0.001 --lambda_1=0.00001 --lambda_2=0.00001 --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True
# python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.1 --learning_rate=0.001 --lambda_1=0.4 --lambda_2=0.4 --parameter_sharing=True --use_ordinal_cost=True --normalize_input=False
# python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.05 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0007 --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True
# python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.1 --learning_rate=0.001 --lambda_1=0.0 --lambda_2=0.0 --parameter_sharing=True --use_ordinal_cost=True --normalize_input=False --use_user_vector=True

python3 -m cfnade.trainer --dataset=movielens-100k --learning_rate=0.001 --lambda_w=0.009 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=False --use_user_vector=True --seed=1
