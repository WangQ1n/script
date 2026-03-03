import sqlite3
import struct


keys = ["open_left", "open_right", "closed", "open", "sleep", "car1_active", "car6_active", "zero", "car1_forward", "car6_forward", "car1_det", "car6_det", "x", "xx", "xxx", "xxxx"]


def parse_entry(data: bytes):
    """解析一条二进制数据"""
    result = {}

    # 0,1 -> uint8
    # result["byte0"] = data[0]
    # result["byte1"] = data[1]

    # 2 -> bool[8] (bit flags)
    for i in range(8):
        result[keys[i]] = bool((data[2] >> i) & 1)
    # 3~8 -> 6 * uint8
    # result["bytes3_8"] = list(data[3:9])

    # 9 -> bool[8] (bit flags)
    for i in range(8):
        result[keys[i+8]] = bool((data[9] >> i) & 1)

    # 10~19 -> 5 * uint16 (小端)
    u16_values = struct.unpack(">5H", data[10:20])
    result["speed"] = u16_values[0] * 0.1
    result["next_station"] = u16_values[1]
    result["destination"] = u16_values[2]
    # result["u16_16_17"] = u16_values[3]
    # result["u16_18_19"] = u16_values[4]

    # 20 -> uint8
    # result["byte20"] = data[20]

    return result


def main():
    # 1. 连接到 db 文件
    db_path = "/home/crrcdt123/windows_desktop/苏八运行数据/0806-2025.10.26/10/db/dbtcms.db"
    conn = sqlite3.connect(db_path)

    # 2. 获取游标
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parsed (
        ts TEXT,
        col1 INTEGER,
        col2 INTEGER,
        open_left INTEGER, open_right INTEGER, closed INTEGER, open INTEGER,
        sleep INTEGER, car1_active INTEGER, car6_active INTEGER, zero INTEGER,
        car1_forward INTEGER, car6_forward INTEGER, car1_det INTEGER, car6_det INTEGER,
        x INTEGER, xx INTEGER, xxx INTEGER, xxxx INTEGER,
        speed INTEGER, next_station INTEGER, destination INTEGER
    )
    """)

    # 3. 执行 SQL 查询
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("所有表：", tables)

    # 4. 查询某个表的数据 ComData
    cursor.execute("SELECT * FROM ComData;")
    rows = cursor.fetchall()
    
    for ts, col1, col2, raw in rows:
        parsed = parse_entry(raw)
        print(f"时间: {ts}, col1={col1}, col2={col2}")
        print("解析结果:", parsed)
        print("-" * 50)
        # 插入到新表
        cursor.execute("""
            INSERT INTO parsed VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
        """, (
            ts, col1, col2,
            parsed["open_left"], parsed["open_right"], parsed["closed"], parsed["open"],
            parsed["sleep"], parsed["car1_active"], parsed["car6_active"], parsed["zero"],
            parsed["car1_forward"], parsed["car6_forward"], parsed["car1_det"], parsed["car6_det"],
            parsed["x"], parsed["xx"], parsed["xxx"], parsed["xxxx"],
            parsed["speed"], parsed["next_station"], parsed["destination"]
        ))
    conn.commit()
    # 5. 关闭连接
    conn.close()
    print("✅ 数据解析并保存到 parsed_logs 完成！")


if __name__ == "__main__":
    main()