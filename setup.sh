# install packages
pip install lxml simplejson pyyaml nltk
pip install spacy==2.1.8 word2number
python -m spacy download en
# clone repo and download dataset
git clone https://github.com/izumi-h/ccgcomp.git
cd ccgcomp
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
./tools/install_tools.sh
./tools/install_parsers.sh 
# extract data
mkdir plain_multinli_train
sh tools/extract_mnli_train.sh
# apply parser
mkdir LF
mkdir mnli_train_results
# arg1: start number, argv2: end number
python process_mnli.py 1 10000
