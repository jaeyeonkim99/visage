CFG_PATH="cfg/visage_vggsound.yaml"

accelerate launch --main_process_port=1234 train_mono.py $CFG_PATH