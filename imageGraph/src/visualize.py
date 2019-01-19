asin ={}
import array
i = 0
with open("../data/temp.b", "rb") as in_file:
    while (i<100):
        piece = in_file.read(10)
        a = array.array('f')
        a.fromfile(in_file,4096)
        i+=1
        asin[int.from_bytes(piece,byteorder="big")]= a.tolist()
        if piece == "":
            break # end of file
import json
print (type(a.tolist()),type(a.tolist()[0]))
with open('temp_100.json', 'w') as fp:
    json.dump(asin, fp)