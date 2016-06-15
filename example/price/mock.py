import random, math

series = [i+1 for i in range(200)]

series[30] = 1
series[60] = 1

for i in range(10000):
    k = random.randint(0, 199)
    count = 1000 / series[k]
    if k == 30:
        count = 0
    for j in range(count):
        dis = random.random() * 10
        price = series[k] / math.sqrt(1.0 + dis)
        print str(price) + '\t' + str(dis) + '\t' + str(k)
