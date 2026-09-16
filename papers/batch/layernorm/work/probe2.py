import subprocess
cmd = r"""
R=/Users/vzhu/Developer/research-copy
ls $R
echo --- data_cache
ls $R/data_cache 2>/dev/null | head -30
echo --- mnist search
find $R -maxdepth 6 -iname '*MNIST*' 2>/dev/null | head -20
echo --- idx files
find $R -maxdepth 8 -name '*idx3-ubyte*' 2>/dev/null | head
echo --- sibling work dirs
ls $R/runs/batch/batchnorm/work 2>/dev/null | head -20
echo --- net test
python -c "import urllib.request;print(urllib.request.urlopen('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',timeout=20).status)" 2>&1 | tail -2
"""
print(subprocess.run(["bash","-lc",cmd],capture_output=True,text=True).stdout[-4000:])
