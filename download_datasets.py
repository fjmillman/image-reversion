from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import sys
import tarfile
import tempfile
import shutil

dataset = sys.argv[1]

if dataset == 'image-enhancements':
    url = 'https://drive.google.com/open?id=1fQGpMnarbwaZD1I-4Af-DYbm2bz8vfQ8'
elif dataset == 'vignette':
    url = 'https://drive.google.com/open?id=1HyV3JdNVMblKW9AYLe_xDwfZlIREPk7T'
else:
    raise Exception('dataset does not exist')

with tempfile.TemporaryFile() as tmp:
    print("downloading", url)
    shutil.copyfileobj(urlopen(url), tmp)
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp)
    tar.extractall()
    tar.close()
    print("done")
