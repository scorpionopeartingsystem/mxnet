import sys
from operator import itemgetter

data = {}
for line in sys.stdin:
    price, deal_price, create_time, car_id, minor_category_name, \
        license_date, license_month, road_haul, city_id, \
        pinpai, chexi, chexing, guobie, zhidaojiage, \
        niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, \
        cheshenxingshi = line.strip('\n').split('\t')
    key = pinpai + chexi + chexing + license_date
    if key not in data:
        data[key] = []
    data[key].append((line.strip('\n'), int(create_time)))

sw0 = open("train.t", "w")
sw1 = open("test.t", "w")

sw2 = open("train.tsv", "w")
sw3 = open("test.tsv", "w")
for key, val in data.items():
    val = sorted(val, key = itemgetter(1))
    for i in range(len(val)):
        start = max(0, i - 5)
        new_line = '||'.join([line for line, tm in val[start:(i+1)]])
        if val[i][1] > 1462032000:
            sw1.write(new_line + '\n')
            sw3.write(val[i][0] + '\n')
        else:
            sw0.write(new_line + '\n')
            sw2.write(val[i][0] + '\n')
sw0.close()
sw1.close()

