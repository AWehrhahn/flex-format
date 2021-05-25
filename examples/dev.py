import numpy as np
import pandas as pd
from flex.flex import FlexFile
from flex.extensions.bindata import BinaryDataExtension, MultipleDataExtension
from flex.extensions.tabledata import TableExtension, JSONTableExtension


file = FlexFile()

ext_bin = BinaryDataExtension(data=np.linspace(0, 1, 9).reshape((3,3)))

ext = MultipleDataExtension()
ext.data["data"] = np.arange(12, dtype="i2").reshape((3, 4))

tab = JSONTableExtension()
tab.data = pd.DataFrame(np.arange(20).reshape((10, 2)), columns=["A", "B"])

file.header["bla"] = np.float(3e10)
file.header["blub"] = np.nan
file.header["blrub"] = float("inf")
file.extensions["np"] = ext
file.extensions["tab"] = tab
file.extensions["bin"] = ext_bin

d = file.to_json()
d2 = FlexFile.from_json(d)

d = file.to_json("test.json")
d2 = FlexFile.from_json("test.json")

d = file.to_fits("test.fits", overwrite=True)
d2 = FlexFile.from_fits("test.fits")

file.write("test.flex")
f2 = FlexFile.read("test.flex")

f2["np"].data["data"][:5] = 1
print(f2["np"].data)

del f2
f2 = FlexFile.read("test.flex")

print(f2["np"].data)
