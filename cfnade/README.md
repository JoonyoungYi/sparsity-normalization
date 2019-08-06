python3 -m cfnade.trainer --dataset=movielens-100k --learning_rate=0.0001 --lambda_w=0.01 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=False --use_user_vector=True --seed=1
python3 -m cfnade.trainer --dataset=movielens-100k --learning_rate=0.0001 --lambda_w=0.009 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=True --use_user_vector=True --seed=1

python3 -m cfnade.trainer --dataset=movielens-100k --learning_rate=0.0001 --lambda_w=0.005 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=False --use_user_vector=False --seed=1
python3 -m cfnade.trainer --dataset=movielens-100k --learning_rate=0.0001 --lambda_w=0.04 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=True --use_user_vector=False --seed=1

python3 -m cfnade.trainer --dataset=movielens-1m --learning_rate=0.0001 --lambda_w=0.004 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=False --use_user_vector=True --seed=1
python3 -m cfnade.trainer --dataset=movielens-1m --learning_rate=0.0001 --lambda_w=0.007 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=True --use_user_vector=True --seed=1

python3 -m cfnade.trainer --dataset=movielens-1m --learning_rate=0.0001 --lambda_w=0.005 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=False --use_user_vector=False --seed=1
python3 -m cfnade.trainer --dataset=movielens-1m --learning_rate=0.0001 --lambda_w=0.009 --parameter_sharing=False --use_ordinal_cost=True --normalize_input=True --use_user_vector=False --seed=1
