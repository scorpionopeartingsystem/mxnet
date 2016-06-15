#coding=utf-8

import re
import MySQLdb

def conn():
    rconn = re.compile(r"reco_db = ([a-z0-9]+):([a-z0-9]+)@tcp\((.+):(\d+)\)/(.+)")
    db = None
    with open("/etc/reco.conf") as f:
        for line in f:
            match = re.search(rconn,line)
            if match:
                user,password,ip,port,dbname = match.group(1),match.group(2),match.group(3),match.group(4),match.group(5)
                db = MySQLdb.connect(ip,user,password,dbname,port = int(port),charset = 'utf8')
                break
            else:
                print "get connect string error"
    return db

if __name__ == "__main__":
    conn()
