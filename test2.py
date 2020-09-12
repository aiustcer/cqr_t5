from tqdm import tqdm
a = {"i":1}
b = {'i':2}

list = [a,b]


for i in list:
    i.update({'j':3})