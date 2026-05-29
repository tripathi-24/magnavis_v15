import time
import mysql.connector
from data_convert_db_now import DB_CONFIG, TABLE_NAME

def run_query(q, params):
    t0 = time.time()
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(q, params)
        res = cur.fetchall()
        t1 = time.time()
        print(f"Time: {t1-t0:.4f}s | Query: {q} | Res: {res[:2]}")
    except Exception as e:
        print(f"Err on {q}: {e}")
    finally:
        try:
            conn.close()
        except: pass

sid = "OBS2_1"
run_query(f"SELECT timestamp FROM {TABLE_NAME} WHERE sensor_id=%s AND id > (SELECT MAX(id) - 5000000 FROM {TABLE_NAME}) ORDER BY id DESC LIMIT 1", (sid,))
run_query(f"SELECT DISTINCT sensor_id FROM {TABLE_NAME} WHERE id > (SELECT MAX(id) - 500000 FROM {TABLE_NAME}) ORDER BY id DESC LIMIT 6", ())
