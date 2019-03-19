#!/pwrlocal/pub/anaconda/py3/bin/python3

import os
import sys
from os.path import join

def main():
  path = os.environ.get('PROJECTDIR')
  template, *args_cmd = sys.argv[1:]
  with open(join(path, template), 'r') as f:
    content = f.read()
    print(content.format(*args_cmd))

if __name__ == "__main__":
  main()

