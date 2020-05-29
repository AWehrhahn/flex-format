import numpy as np
import pandas as pd
from fits2.fits2 import Fits2File
from fits2.extensions.bindata import BinaryDataExtension
from fits2.extensions.tabledata import TableExtension, JSONTableExtension


file = Fits2File()
ext = BinaryDataExtension()
ext.data = np.zeros(12, dtype="f2")

tab = JSONTableExtension()
tab.data = pd.DataFrame(np.ones((10, 2)), columns=["A", "B"])


file.header["bla"] = "blub"
file.extensions["np"] = ext
file.extensions["tab"] = tab

file.write("test.fits2")

f2 = Fits2File.read("test.fits2")

f2["np"].data[:5] = 1
print(f2["np"].data)

del f2
f2 = Fits2File.read("test.fits2")

print(f2["np"].data)
