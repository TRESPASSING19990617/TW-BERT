unset $(/usr/bin/env | egrep '^(\w+)=(.*)$' | egrep ^'DSW' | /usr/bin/cut -d= -f1);
cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/msvd_qa.json'

TXT_DB='data/msvd_qa/txt/test.jsonl'
IMG_DB='/datasets/MSVD/MSVD_Videos'

horovodrun -np 8 python src/tasks/run_video_qa.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir  \
      --config $CONFIG_PATH
