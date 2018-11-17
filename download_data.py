import urllib

testfile = urllib.URLopener()
testfile.retrieve("https://www.dropbox.com/s/kopsuw53kv9xl2n/mnist_train.csv?dl=0", "mnist.csv")