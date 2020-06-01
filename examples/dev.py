import numpy as np
import pandas as pd
from flex.flex import FlexFile
from flex.extensions.bindata import BinaryDataExtension, MultipleDataExtension
from flex.extensions.tabledata import TableExtension, JSONTableExtension


file = FlexFile()
ext = MultipleDataExtension()
ext.data["data"] = np.zeros(12, dtype="f2")

tab = JSONTableExtension()
tab.data = pd.DataFrame(np.ones((10, 2)), columns=["A", "B"])


file.header["bla"] = "blub"
file.extensions["np"] = ext
file.extensions["tab"] = tab

file.write("test.fits2")

f2 = FlexFile.read("test.fits2")

f2["np"].data["data"][:5] = 1
print(f2["np"].data)

del f2
f2 = FlexFile.read("test.fits2")

print(f2["np"].data)
