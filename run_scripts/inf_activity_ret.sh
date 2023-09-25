unset $(/usr/bin/env | egrep '^(\w+)=(.*)$' | egrep ^'DSW' | /usr/bin/cut -d= -f1);
cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/activity_ret.json'

TXT_DB='data/activity_ret/txt/test.jsonl'
IMG_DB='/datasets/ActivityNet/Activity_Videos_Compress'
a='best'

horovodrun -np 8 python src/tasks/run_video_retrieval.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $a \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir  \
      --config $CONFIG_PATH
