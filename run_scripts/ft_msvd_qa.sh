unset $(/usr/bin/env | egrep '^(\w+)=(.*)$' | egrep ^'DSW' | /usr/bin/cut -d= -f1);
cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/msvd_qa.json'

horovodrun -np 8 python src/tasks/run_video_qa.py \
      --config $CONFIG_PATH \
      --output_dir ./results/alpro_finetune/msvd_qa/$(date '+%Y%m%d%H%M%S')
