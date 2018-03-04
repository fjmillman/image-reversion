FILE=$1
URL=https://drive.google.com/file/d/1BnAsqCu4nbi0E0b0htvOAIpurGUYwBCc
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
