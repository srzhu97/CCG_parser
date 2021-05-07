# install packages

pip install lxml simplejson pyyaml nltk
pip install spacy==2.1.8 word2number
python -m spacy download en
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
chmod +x ./tools/install_tools.sh
./tools/install_tools.sh
chmod +x ./tools/install_parsers.sh 
./tools/install_parsers.sh linux
# extract data
mkdir plain_multinli_train
sh tools/extract_mnli_train.sh
# apply parser
mkdir cache
mkdir mnli_train_results
chmod +x ./scripts/eval_mnli_train.sh
chmod +x ./scripts/rte_train.sh
# arg1: start number, argv2: end number
mkdir LF
python process_mnli.py 1 5
