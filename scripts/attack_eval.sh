

# python3 code/eval_under_attack.py \
#   --config_file=config/config_cw_with_noise.yaml \
#   --config_name=attack
# 
# python3 code/eval_under_attack.py \
#   --config_file=config/config_cw_without_noise.yaml \
#   --config_name=attack

# python3 code/eval_under_attack_v2.py \
#   --config_file=config/config_cifar_fgm_with_noise.yaml \
#   --config_name=attack
# 
# python3 code/eval_under_attack_v2.py \
#   --config_file=config/config_cifar_fgm_without_noise.yaml \
#   --config_name=attack

python3 code/eval_under_attack.py \
  --config_file=config/config_cifar_with_noise.yaml \
  --config_name=attack

python3 code/eval_under_attack.py \
  --config_file=config/config_cifar_without_noise.yaml \
  --config_name=attack
