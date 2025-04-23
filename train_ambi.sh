CFG_PATH="cfg/visage_ytambigen.yaml"

accelerate launch --main_process_port=5231 train_ambi.py $CFG_PATH