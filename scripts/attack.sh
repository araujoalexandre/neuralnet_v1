
FOLDER="/pwrwork/rech/jvd/ubx31mc/neuralnet/circulant/cifar10_dense"
DATE="$1"
TRAIN_DIR="${FOLDER}/${DATE}"
LOGS_DIR="${TRAIN_DIR}_logs"
CONFIG_FILE="${LOGS_DIR}/model_flags.yaml"

# python3 code/eval_under_attack.py \
#        	--config_file=${CONFIG_FILE} \
# 	--config_name=attack \
# 	--train_dir=${TRAIN_DIR} \
#         &>> ${LOGS_DIR}/log_carlini.logs & 

python3 code/eval_under_attack_v2.py \
       	--config_file=${CONFIG_FILE} \
	--config_name=attack \
	--train_dir=${TRAIN_DIR} \
        &>> ${LOGS_DIR}/log_fgm_inf.logs & 
