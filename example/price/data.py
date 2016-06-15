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

car_source = fetch("SELECT a.price, a.deal_price, a.create_time, a.car_id, a.minor_category_name, a.license_date, a.license_month, a.road_haul, a.city_id, b.pinpai, b.chexi, b.chexing, b.guobie, b.zhidaojiage, b.niankuan, b.cheliangleixing, b.cheliangjibie, b.biansuxiangleixing, b.cheshenxingshi FROM ganji_vehicle.vehicle_c2c_car_source a join ganji_vehicle.liyang_car b where a.car_id = b.id;")

for row in car_source:
    print '\t'.join([str(x).strip() for x in row])
