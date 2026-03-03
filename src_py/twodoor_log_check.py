import re
import sys
import os
import csv
from datetime import datetime, timedelta

# 二门日志检查
# 线路站点完整性检查，是否存在漏站，重复
# ======================
# 配置
# ======================

TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"

LOG_PATTERN = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*?'
    r'train in station (\d+) is closed'
)

# ======================
# 读取 station 顺序
# ======================

def load_station_sequence(txt_path):
    seq = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("：", ":")
            if not line or line.startswith("#"):
                continue
            _, sid = line.split(":", 1)
            seq.append(int(sid.strip()))
    return seq

# ======================
# 解析日志
# ======================

def parse_log(log_file, target_date):
    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue

            ts = datetime.strptime(m.group(1), TIME_FMT)
            if ts.date() != target_date:
                continue

            records.append({
                "time": ts,
                "station": int(m.group(2))
            })

    records.sort(key=lambda x: x["time"])
    return records

# ======================
# 连续重复去重
# ======================

def dedup_consecutive(seq):
    if not seq:
        return []
    result = [seq[0]]
    for sid in seq[1:]:
        if sid != result[-1]:
            result.append(sid)
    return result

# ======================
# 按方向切分 segment
# ======================

def which_direction(station, up_set, down_set):
    if station in up_set and station in down_set:
        return "BOTH"
    if station in up_set:
        return "UP"
    if station in down_set:
        return "DOWN"
    return "UNKNOWN"

def split_by_membership(records, up_seq, down_seq):
    up_set = set(up_seq)
    down_set = set(down_seq)

    segments = []
    cur_dir = None
    cur_seg = []

    for r in records:
        sid = r["station"]
        time = r["time"]
        d = which_direction(sid, up_set, down_set)

        if cur_dir is None:
            cur_dir = d
            cur_seg = [{"time": time, "station": sid}]
        elif (d == cur_dir or d == "BOTH") and (time - cur_seg[-1]['time']).total_seconds() // 60 < 60:
            cur_seg.append({"time": time, "station": sid})
        else:
            segments.append((cur_dir, cur_seg))
            cur_dir = d
            cur_seg = [{"time": time, "station": sid}]

    if cur_seg:
        segments.append((cur_dir, cur_seg))

    return segments

def check_station_order(actual_seq, expected_seq):
    expected_index = {sid: i for i, sid in enumerate(expected_seq)}
    issues = []
    last_idx = None

    for pos, sid in enumerate(actual_seq):
        if sid not in expected_index:
            issues.append({
                "pos": pos,
                "station": sid,
                "type": "UNKNOWN"
            })
            continue

        cur_idx = expected_index[sid]

        if last_idx is not None:
            if cur_idx < last_idx:
                issues.append({
                    "pos": pos,
                    "station": sid,
                    "type": "REVERSED"
                })
            elif cur_idx > last_idx + 1:
                issues.append({
                    "pos": pos,
                    "station": sid,
                    "type": "SKIPPED",
                    "expected": expected_seq[last_idx + 1]
                })
            elif cur_idx == last_idx:
                issues.append({
                    "pos": pos,
                    "station": sid,
                    "type": "REPEATED",
                    "expected": expected_seq[last_idx + 1 if last_idx + 1 < len(expected_index) else last_idx]
                })

        last_idx = cur_idx
    if len(actual_seq) < len(expected_seq):
        issues.append({
                    "pos": 0,
                    "station": 0,
                    "type": "INCOMPLETE",
                    "expected": 0
                })
    return issues

# ======================
# 保存结果
# ======================
def fmt_minute(t: datetime):
    return t.strftime("%Y-%m-%d %H:%M")

def save_report(segments, up_seq, down_seq, out_path, mode="w"):
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id", "start_time", "end_time", "cost_time", "direction", "station_sequence", "issue_type",
            "position", "station", "expected_station", "station_count"
        ])

        for seg_id, (direction, record) in enumerate(segments):
            expected = up_seq if direction == "UP" else down_seq
            seq = [r["station"] for r in record]
            times = [r["time"] for r in record]
            issues = check_station_order(seq, expected)
            duration = (times[-1] - times[0]).total_seconds() // 60

            if not issues:
                writer.writerow([seg_id, fmt_minute(times[0]), fmt_minute(times[-1]), duration, direction, seq, "OK", "", "", "", len(seq)])
            else:
                for it in issues:
                    writer.writerow([
                        seg_id,
                        fmt_minute(times[0]),
                        fmt_minute(times[-1]),
                        duration,
                        direction,
                        seq,
                        it["type"],
                        it["pos"],
                        it["station"],
                        it.get("expected", ""),
                        len(seq)
                    ])


def gen_date_list(start_date: str, days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(days)
    ]

def main():
    log_file = "/media/crrcdt123/glam/crrc/data/su8/2door/0818/clear/anti_clamp.log"   # 改成你的日志路径
    station_up_file = "/media/crrcdt123/glam/crrc/data/su8/2door/0818/clear/stationID-up.txt"
    station_down_file = "/media/crrcdt123/glam/crrc/data/su8/2door/0818/clear/stationID-down.txt"
    out_file = "/media/crrcdt123/glam/crrc/data/su8/2door/0818/clear/result.csv"
    if os.path.exists(out_file):
        os.remove(out_file)
    up_seq = load_station_sequence(station_up_file)
    up_seq.reverse()
    down_seq = load_station_sequence(station_down_file)
    datatimes = gen_date_list("2026-01-12", 12)
    # datatimes = ["2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", 
    #              "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19",
    #              "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23"]
    print(datatimes)
    records = []
    for datatime in datatimes:
        target_date = datetime.strptime(datatime, "%Y-%m-%d").date()
        records.append(parse_log(log_file, target_date))
    records = [x for row in records for x in row]
    if not records:
        print("No matched log records.")
        return

    # station_seq = [r["station"] for r in records]
    # station_seq = dedup_consecutive(station_seq)

    segments = split_by_membership(records, up_seq, down_seq)

    os.makedirs("output", exist_ok=True)
    save_report(segments, up_seq, down_seq, out_file, 'a')

    print("Analysis finished.")
    print("Segments:", len(segments))
    print("Result saved to:", out_file)


if __name__ == "__main__":
    main()
