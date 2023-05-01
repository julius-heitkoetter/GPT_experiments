import os
import sys
import tiktoken
import numpy as np

assert len(sys.argv)==2, "ERROR: need to pass in direcory to prepare data from"
input_path = sys.argv[1]
print("Preparing data in input directory ", input_path)
print("")

data = ""
for filename in os.listdir(input_path):
   with open(os.path.join(input_path, filename), 'r') as f:
    print("Opened ", filename)
    data += f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(input_path, 'train.bin'))
val_ids.tofile(os.path.join(input_path, 'val.bin'))