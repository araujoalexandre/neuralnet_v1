PROCESS=$(pgrep python3)
for pid in ${PROCESS}
do
  kill -9 ${pid}
done

