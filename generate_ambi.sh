CKPT=""
BATCH_SIZE=""
SAVE_PATH=""
TEST_CSV="csv/yt_ambigen/test.csv"

python generate_ambi.py -s $SAVE_PATH -c $CKPT -b $BATCH_SIZE -tc $TEST_CSV