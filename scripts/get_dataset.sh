OUTPUT_DIR_PATH=$1
FILE_ZIP="dataset.zip"

wget -O $FILE_ZIP "https://www.grip.unina.it/download/prog/AdversarialRobustnessCLIP/AdversarialRobustnessCLIP.zip"
unzip $FILE_ZIP -d $OUTPUT_DIR_PATH && rm $FILE_ZIP