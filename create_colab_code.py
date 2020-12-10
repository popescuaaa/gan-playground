from os import listdir
from os.path import isfile, join
import sys

if __name__ == '__main__':
    folder_name = sys.argv[1]
    outfile = sys.argv[2]

    files = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and f.split('.')[-1] == 'py']
    _files = [folder_name + f for f in files]
    print(_files)
    r = open(outfile, 'w+')
    for f in _files:
        file = open(f)
        for line in file:
            if 'from' not in line and 'import' not in line:
                r.write(line)
        file.close()
    r.close()
    print('the result is in: {}'.format(outfile))
