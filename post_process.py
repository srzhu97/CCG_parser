import os
from os import listdir
from os.path import isfile, join

def main():
    # use only depccg result
    path = "./LF"
    parser_name = 'easyccg'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if parser_name not in file:
            continue
        id = file.split("-")[2]
        with open(file, 'r') as f:
            lf_forms = f.readlines()
        

if __name__ == '__main__':
    main()