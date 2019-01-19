import json
import struct

with open('../data/data.json') as json_data:
    d = json.load(json_data)
f = open("temp.b","wb")
for key in d :
    f.write(int(key).to_bytes(10,byteorder="big"))
    for i in d[key]:
        f.write(bytearray(struct.pack("f", i)))
f.close()