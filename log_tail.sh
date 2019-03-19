log_folder=$(find ${WORKDIR}/models/* -type d -iname "*logs" | sort | tail -n 1)
if [ "$#" -ne 2 ]
then
  tail -f ${log_folder}/log_$1.logs
else
  tail -f ${log_folder}/log_$1.logs --lines=$2
fi
