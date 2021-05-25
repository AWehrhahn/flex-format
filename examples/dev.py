import numpy as np
import pandas as pd
from flex.flex import FlexFile
from flex.extensions.bindata import BinaryDataExtension, MultipleDataExtension
from flex.extensions.tabledata import TableExtension, JSONTableExtension


file = FlexFile()

ext_bin = BinaryDataExtension(data=np.linspace(0, 1, 4*3*2).reshape((4,3,2)))

ext = MultipleDataExtension()
ext.data["data"] = np.arange(12, dtype="i2").reshape((3, 2, 2))

tab = JSONTableExtension()
tab.data = pd.DataFrame(np.arange(20).reshape((10, 2)), columns=["A", "B"])

file.header["bla"] = np.float(3e10)
file.header["blub"] = np.nan
file.header["blrub"] = float("inf")
file.header["blurb"] = np.array([1, 2, 3])
file.header["blurb2"] = [1, 2, 3]
file.header["int"] = 1
file.header["np_int"] = np.int(1)
file.header["str"] = "Hello"
file.header["np_str"] = np.str("World")
# FITS can handle either long header keys, or long entries
# But not both at the same time...
file.header["really_long_entry"] = "bla"
file.header["really_l"] = "bla" * 100
file.header["really_long_text_with_HIERARCH"] = "bla" * 100

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
