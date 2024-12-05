OUTPUT_DIR_PATH=$1
FILE_ZIP="weights.zip"

wget -O $FILE_ZIP "https://www.grip.unina.it/download/prog/AdversarialRobustnessCLIP/weights_AdversarialRobustnessCLIP.zip"
unzip $FILE_ZIP -d $OUTPUT_DIR_PATH && rm $FILE_ZIP