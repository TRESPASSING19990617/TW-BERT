unset $(/usr/bin/env | egrep '^(\w+)=(.*)$' | egrep ^'DSW' | /usr/bin/cut -d= -f1);
cd ..
  
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/pretrain_TW-BERT.json'

horovodrun -np 8 python src/pretrain/run_pretrain_sparse.py \
      --config $CONFIG_PATH \
      --output_dir ./results/alpro_pt_all_data/$(date '+%Y%m%d%H%M%S')

