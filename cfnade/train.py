# 저자가 했던 실험 리프로듀스.
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.001  --lambda_1=0.015 --lambda_2=0.015 -> 0.8410
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0005 --lambda_1=0.015 --lambda_2=0.015 -> 0.8380
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0002 --lambda_1=0.015 --lambda_2=0.015 -> 0.8329
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.001  --lambda_1=0.02 --lambda_2=0.02 -> 0.8434
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0005 --lambda_1=0.02 --lambda_2=0.02 -> 0.8385
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0002 --lambda_1=0.02 --lambda_2=0.02 -> 0.8359
# 결론: seed=1에 대해서 대충 0.8329 나오니까 잘 나오는 것!

# 저자가 했던 실험에 더해 seed=1에 대해서 weight decay 쓸 때 얼마나 안좋아지는지 확인.
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.001  --lambda_1=0.0  --lambda_2=0.0 -> 0.8674
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0005 --lambda_1=0.0  --lambda_2=0.0 -> 0.8632
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0002 --lambda_1=0.0  --lambda_2=0.0 -> 0.8586
# 결론: 엄청 안좋아지지는 않고 조금 안좋아짐. 아마도 parameter sharing과 ordinal cost의 결과일 것! 그래서 확인해보기로 함.

# ordinal cost의 영향에 대해서 확인 (진행중)
# [TODO] python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.001  --lambda_1=0.0 --lambda_2=0.0 --use_ordinal_cost=False
# [TODO] python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0005 --lambda_1=0.0 --lambda_2=0.0 --use_ordinal_cost=False
# [TODO] python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0002 --lambda_1=0.0 --lambda_2=0.0 --use_ordinal_cost=False

# parameter sharing의 영향에 대해서 확인 (진행중)
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.001  --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False -> 0.8777
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0005 --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False -> 0.8742
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0002 --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False -> 0.8728

# parameter sharing, ordinal cost둘 다 없을 때 어떻게 되는지 확인하기. 영향에 대해서 확인 (진행중)
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0002 --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False --use_ordinal_cost=False -> 0.8942

# 아래 실험들은 다시 짜서 다시 돌려야 함! normalize_input 들어간 실험들은 잘못했단 것을 깨달음!
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.001  --lambda_1=0.0  --lambda_2=0.0  --normalize_input=True -> 1.0401
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0005 --lambda_1=0.0  --lambda_2=0.0  --normalize_input=True -> 1.0399
# python3 -m cfnade.trainer --gpu_memory_fraction=0.25 --learning_rate=0.0002 --lambda_1=0.0  --lambda_2=0.0  --normalize_input=True -> 1.0365
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0005 --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False --normalize_input=True -> 1.0395
# python3 -m cfnade.trainer --gpu_memory_fraction=0.24 --learning_rate=0.0002 --lambda_1=0.0  --lambda_2=0.0 --parameter_sharing=False --normalize_input=True --use_ordinal_cost=False -> 1.1169
