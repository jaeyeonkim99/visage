# Congiuration for generation
CKPT=""
BATCH_SIZE=""
TEST_CSV="csv/yt_ambigen/test.csv"
SAVE_PATH=""

# Congiruation for evaluation
TEST_SAMPLES=""
RESULT_PATH="" 
GT_AUDIO_MAP_PATH=""

python generate_ambi.py -s $SAVE_PATH -c $CKPT -b $BATCH_SIZE -tc $TEST_CSV

python evaluate_spatial.py -a $SAVE_PATH -r $RESULT_PATH -ga $GT_AUDIO_MAP_PATH

python evaluate_semantic.py -a $SAVE_ID -r $RESULT_PATH -t $TEST_SAMPLES     