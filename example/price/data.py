#coding=utf-8
import math
import MySQLdb
import datetime
import sys
import db_config
reload(sys)
sys.setdefaultencoding('utf-8')


def fetch(sql):
    db = db_config.conn()
    cur = db.cursor()
    res = None
    try:
        cur.execute(sql)
        res = cur.fetchall()
    except Exception, e:
        print e
    return res

car_source = fetch("SELECT a.price, a.deal_price, a.suggest_price, a.evaluator, a.create_time, a.car_id, a.minor_category_name, a.license_date, a.license_month, a.road_haul, a.city_id, b.pinpai, b.chexi, b.chexing, b.guobie, b.zhidaojiage, b.niankuan, b.cheliangleixing, b.cheliangjibie, b.biansuxiangleixing, b.cheshenxingshi FROM ganji_vehicle.vehicle_c2c_car_source a join ganji_vehicle.liyang_car b where a.car_id = b.id;")

good_evaluators = set()
for line in file("./data/good_evaluator.tsv"):
    line = line.strip()
    if len(line) > 0:
        good_evaluators.add(line.strip())

sw_price = open("./data/price.tsv", "w")
sw_deal = open("./data/price_deal.tsv", "w")
sw_eval = open("./data/price_eval.tsv", "w")
for row in car_source:
    row = [str(x).strip() for x in row]
    price, deal_price, suggest_price, evaluator = row[:4]
    others = row[4:]
    sw_price.write('\t'.join([price] + others) + '\n')
    
    price = float(price)
    deal_price = float(deal_price)
    suggest_price = float(suggest_price)
    
    deal_ratio = min(deal_price, price) / (1.0 + max(deal_price, price))
    if deal_price > 0 and deal_ratio > 0.7:
        sw_deal.write('\t'.join([str(deal_price)] + others) + '\n')

    if evaluator in good_evaluators:
        suggest_ratio = min(suggest_price, price) / (1.0 + max(suggest_price, price))
        if suggest_price > 0 and suggest_ratio > 0.7:
            sw_eval.write('\t'.join([str(suggest_price)] + others) + '\n')

sw_price.close()
sw_deal.close()
sw_eval.close()

