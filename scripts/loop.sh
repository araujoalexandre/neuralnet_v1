
PROJECT_PATH="/home/aaraujo/neuralnet_structured_matrices"
GRID_SEARCH_PATH="/srv/osirim/aaraujo/neuralnet_structured_matrices/grid_search_mom_v2"
CONFIG_PATH="${PROJECT_PATH}/config"
CODE_PATH="${PROJECT_PATH}/code"

ID=7
CONFIG_FILE="config_loop_${ID}.yaml"

for i in 30
do
  for j in 0.0001 0.00025	0.0005 0.00075 0.001 0.0025 0.005 0.0075 0.01 0.025 0.05 0.075 0.1
  do

    # setup config
    REPLACE1="s/placeholder_n_layers/${i}/g"
    REPLACE2="s/placeholder_lr/${j}/g"
    sed ${REPLACE1} ${CONFIG_PATH}/config.yaml | \
      sed ${REPLACE2} > ${CONFIG_PATH}/${CONFIG_FILE}
    
    DATE=$(date '+%Y-%m-%d_%H.%M.%S')
    TRAIN_DIR="${GRID_SEARCH_PATH}/${DATE}"
    LOGS_DIR="${GRID_SEARCH_PATH}/${DATE}_logs"
    mkdir ${TRAIN_DIR} ${LOGS_DIR}
    cp ${CONFIG_PATH}/${CONFIG_FILE} ${LOGS_DIR}/model_flags.yaml
    
    # reset cuda
    export CUDA_VISIBLE_DEVICES='';  
    
    python3 ${CODE_PATH}/train.py \
      --config_file=${CONFIG_PATH}/${CONFIG_FILE} \
      --config_name=train \
      --train_dir=${TRAIN_DIR} \
      &>> "${LOGS_DIR}/log_train.logs" &

    python3 ${CODE_PATH}/eval.py \
      --config_file=${CONFIG_PATH}/${CONFIG_FILE} \
      --config_name=eval_test \
      --train_dir=${TRAIN_DIR} \
      &>> "${LOGS_DIR}/log_eval_test.logs" &

    wait

    python3 ${CODE_PATH}/parse_events.py \
      --folder=${GRID_SEARCH_PATH}/${DATE}
  
  done
done



