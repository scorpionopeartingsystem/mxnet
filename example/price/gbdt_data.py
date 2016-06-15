import sys, datetime
from operator import itemgetter

def process_line(line):
    price, deal_price, car_id, minor_category_name, \
        license_date, license_month, road_haul, city_id, \
        pinpai, chexi, chexing, guobie, zhidaojiage, \
        niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, \
        cheshenxingshi = line.strip('\n').split('\t')
    deal_price = float(deal_price)
    price = float(price)
    ratio = min(price, deal_price) / max(price, deal_price)
    date_delta = datetime.date.today() - datetime.date(int(license_date),int(license_month),15)
    vehicle_age = date_delta.days
    pinpai = pinpai.replace(" ", "").replace(":", "")
    chexi = chexi.replace(" ", "").replace(":", "")
    chexi = pinpai + "_" + chexi
    chexing = chexing.replace(" ", "").replace(":", "")
    chexing = chexi + "_" + chexing
    vehicle_age /= 365.0
    road_haul = float(road_haul) / 10000.0
    zhidaojiage_tks = zhidaojiage.split("-")
    zhidaojiage_min = 0
    zhidaojiage_max = 0
    try:
        zhidaojiage_min = float(zhidaojiage_tks[0])
        zhidaojiage_max = float(zhidaojiage_tks[-1])
    except Exception, e:
        pass
    return price / 10000.0, vehicle_age, 'pp_' + pinpai, 'chexi_' + chexi, 'chexing_' + chexing, road_haul, city_id, \
        zhidaojiage_min, zhidaojiage_max, 'gb_' + guobie, 'nk_' + niankuan, 'cllx_' + cheliangleixing, 'cljb_' + cheliangjibie, 'bsxlx_' + biansuxiangleixing, 'csxs_' + cheshenxingshi


def insert_vocab(a, b):
    if b not in a:
        a[b] = len(a) + 5

def build_vocab(fname):
    ret = {}
    for line in file(fname):
        price, vehicle_age, pinpai, chexi, chexing, road_haul, city_id, \
            zhidaojiage_min, zhidaojiage_max, guobie, niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, cheshenxingshi = process_line(line)
        insert_vocab(ret, pinpai)
        insert_vocab(ret, chexi)
        insert_vocab(ret, chexing)
        insert_vocab(ret, guobie)
        insert_vocab(ret, niankuan)
        insert_vocab(ret, cheliangleixing)
        insert_vocab(ret, cheliangjibie)
        insert_vocab(ret, biansuxiangleixing)
        insert_vocab(ret, cheshenxingshi)
        insert_vocab(ret, city_id)
    return ret

vocab = build_vocab("price0.tsv")

fname = sys.argv[1]
sw = open(fname + ".svm", "w")
for line in file(fname):
    price, vehicle_age, pinpai, chexi, chexing, road_haul, city_id, \
        zhidaojiage_min, zhidaojiage_max, guobie, niankuan, cheliangleixing, \
        cheliangjibie, biansuxiangleixing, cheshenxingshi = process_line(line)
    output = {
        1: vehicle_age,
        2: road_haul,
        3: zhidaojiage_min,
        4: zhidaojiage_max,
        vocab[city_id]: 1,
        vocab[guobie]: 1,
        vocab[pinpai]: 1,
        vocab[chexi]: 1,
        vocab[chexing]: 1,
        vocab[niankuan]: 1, 
        vocab[cheliangleixing]: 1, 
        vocab[cheliangjibie]: 1,
        vocab[biansuxiangleixing]: 1, 
        vocab[cheshenxingshi]: 1
    }
    output = sorted(output.items(), key=itemgetter(0))
    output_line = str(price) + "\t" + "\t".join([str(x) + ":" + str(y) for x, y in output])
    sw.write(output_line + "\n")
sw.close()
    
