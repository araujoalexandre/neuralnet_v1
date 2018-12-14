  
log_folder=$(find ${WORKDIR}/* -type d -iname "*logs" | sort | tail -n 1)
if [ "$#" -ne 1 ]
then
  tail -f ${log_folder}/log_eval_test.logs
else
  tail -f ${log_folder}/log_eval_test.logs --lines=$1
fi
