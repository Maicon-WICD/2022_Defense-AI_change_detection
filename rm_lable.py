import os
f3 = open("label_3_list.txt", 'r')
count = 0
while True:
    print(count)    
    line = f3.readline()
    count += 1
    if count < 9500:
        if os.path.isfile(line[:-1]) and os.path.isfile(line[:13] + 'x/'+line[15:-1]):
            os.remove(line[:-1])
            os.remove(line[:13] + 'x/'+line[15:-1])
        else:
            continue
    else:
        break
    print(count)
    
f3.close()