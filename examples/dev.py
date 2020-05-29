import numpy as np
from fits2.fits2 import Fits2File
from fits2.extensions.npdata import NpDataExtension


file = Fits2File()
ext = NpDataExtension()
ext.data = np.zeros(10)

file.header["bla"] = "blub"
file.extensions["np"] = ext

file.write("test.fits2")

f2 = Fits2File.read("test.fits2")

print(f2.header)
