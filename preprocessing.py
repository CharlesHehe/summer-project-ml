from numpy import unique
from numpy import savez_compressed
import os
import struct

struct_fmt = 'i' + 'f' * 6  # int[5], float, byte[255]
struct_len = struct.calcsize(struct_fmt)
struct_unpack = struct.Struct(struct_fmt).unpack_from
results = []
train_data = []

files = os.listdir('./raw_data')

for file in files:
    with open(f"./raw_data/{file}", "rb") as f1:
        while True:
            data = f1.read(28)
            if not data:
                break
            s = list(struct_unpack(data))
            train_data.append(s)

train_data = unique(train_data, axis=0)

savez_compressed('train_data.npz', train_data)
