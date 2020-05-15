#BSUB -J train
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -n 8
#BSUB -R "span[ptile=1]"
#BSUB -W 1200
#BSUB -o /linkhome/rech/grpgen/ubx31mc/neuralnet/job.%J.train.out
#BSUB -e /linkhome/rech/grpgen/ubx31mc/neuralnet/job.%J.train.err

# DATE=$(date '+%Y-%m-%d_%H.%M.%S')
# DATE=${DATE}.$(date +"%4N")
DATE="2019-05-16_13.35.28.1982"

WORKER_PORT=2222
PS_PORT=2223

INDEX=0
TF_WORKER=""
TF_PS=""
for HOST in ${LSB_HOSTS}
do

  if [ $INDEX -eq 0 ]
  then
    TF_MASTER=[\"${HOST}:${WORKER_PORT}\"]
  fi

  if [ $(($INDEX % 2)) -eq 0 ]
  then
    TF_PS=${TF_PS}\"${HOST}:${PS_PORT}\",
  fi

  if [ $INDEX -gt 0 ]
  then
    TF_WORKER=${TF_WORKER}\"${HOST}:${WORKER_PORT}\",
  fi
let INDEX=$INDEX+1
done
TF_WORKER=[${TF_WORKER::-1}]
TF_PS=[${TF_PS::-1}] 

function format_tf_config {
   echo "{\"cluster\":{\"master\":${TF_MASTER},\"ps\":${TF_PS},\"worker\":${TF_WORKER}},\"task\":{\"type\":\"$1\", \"index\":$2}, \"environment\":\"cloud\"}"
 }

INDEX_PS=0
INDEX=0
for HOST in ${LSB_HOSTS}
do
  
  if [ $(($INDEX % 2)) -eq 0 ]
  then
    # launch parameters server
    CONFIG=$(format_tf_config ps $INDEX_PS)
    blaunch -no-wait -z $HOST "${PROJECTDIR}/run_imagenet_worker.sh ${INDEX_PS} ps '${CONFIG}' ${DATE}" &
    let INDEX_PS=$INDEX_PS+1
    sleep 2
  fi

  # launch master
  if [ $INDEX -eq 0 ]
  then
    # launch master
    CONFIG=$(format_tf_config master $INDEX)
    blaunch -no-wait -z $HOST "${PROJECTDIR}/run_imagenet_worker.sh ${INDEX} master '${CONFIG}' ${DATE}" &
    sleep 2
  
  # lauch worker
  else
    # launch worker
    let WORKER_INDEX=$INDEX-1
    CONFIG=$(format_tf_config worker $WORKER_INDEX)
    blaunch -no-wait -z $HOST "${PROJECTDIR}/run_imagenet_worker.sh ${WORKER_INDEX} worker '${CONFIG}' ${DATE}" &
  fi
  let INDEX=$INDEX+1
done
 




