log_folder=$(find ${WORKDIR}/* -type d -iname "*logs" | sort | tail -n 1)
tail -f ${log_folder}/log_train.logs
