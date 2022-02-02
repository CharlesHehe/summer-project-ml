from numpy import genfromtxt
from numpy import unique
from numpy import savez_compressed
import numpy as np
import codecs
import binascii

# X = genfromtxt("raw.txt", delimiter=",")
# X = genfromtxt("raw2.txt")
# with open('raw2.txt', encoding='cp1252', 'rb') as f:
# with open('raw2.txt', encoding="utf8", errors='ignore') as f:
#     for line in f:
#         print(line)
# dtype = np.dtype('B')
# with open('raw2.txt', "rb") as f:
# binary_data = f.read()
# text = binary_data.decode('utf-8')
# text = binascii.hexlify(binary_data)
# text = codecs.decode(binary_data, 'base64')
# numpy_data = np.fromfile(f, dtype)
# data = f.read()
# print(numpy_data)

# print(X.shape)
# X = unique(X, axis=0)

# savez_compresseddata.npz', X)('


import struct

struct_fmt = 'i' + 'f' * 6  # int[5], float, byte[255]
struct_len = struct.calcsize(struct_fmt)
struct_unpack = struct.Struct(struct_fmt).unpack_from
results = []
buttons = []
bandsaw = []

# , open("bandsaw.txt", "rb") as f2
with open("training.txt", "rb") as f1, open("bandsaw.txt", "rb") as f2:
    while True:
        data = f1.read(28)
        if not data:
            break
        s = list(struct_unpack(data))
        if s[4:] == [0.0, 0.0, 0.0]:
            s[0] = 0
            buttons.append(s)
        else:
            s[0] = 1
            buttons.append(s)
    while True:
        data = f2.read(28)
        if not data:
            break
        s = list(struct_unpack(data))
        if s[4:] == [0.0, 0.0, 0.0]:
            s[0] = 0
            bandsaw.append(s)
        else:
            s[0] = 1
            bandsaw.append(s)

buttons = unique(buttons, axis=0)
bandsaw = unique(bandsaw, axis=0)
savez_compressed('buttons.npz', buttons)
savez_compressed('bandsaw.npz', bandsaw)
# file = open("raw.txt", "rb")
# while 1:
#     c = struct.unpack('i', file.read(4))
#     f = struct.unpack('f' * 6, file.read(4 * 6))
#
#     print(c, f)
