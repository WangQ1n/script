#!/usr/bin/env python3
import os
import re
import argparse

def delete_files_in_range(folder, start_ts, end_ts, dry_run=False):
    """
    删除文件名中时间戳位于指定范围的文件
    文件名格式: cam113-End-20250926-192259498
    """
    pattern = re.compile(r".*-[A-Za-z]+-\d{8}-(\d+)(?:\..*)?$")
    
    deleted = 0
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if not match:
            continue

        timestamp = int(match.group(1))
        full_path = os.path.join(folder, filename)

        if start_ts <= timestamp <= end_ts:
            if dry_run:
                print(f"[DRY-RUN] Would delete: {full_path}")
            else:
                os.remove(full_path)
                print(f"Deleted: {full_path}")
            deleted += 1

    print(f"\nTotal files {'to delete' if dry_run else 'deleted'}: {deleted}")

if __name__ == "__main__":
    folder = "/media/crrcdt123/glam/crrc/data/su8/2door/0818-20250926/pictures/"
    start = [6300000, 83100000, 103000000, 122800000, 142800000, 162800000, 182800000]
    end = [73500000, 93300000, 113200000, 133000000, 153000000, 173100000, 193000000]
    for idx in range(len(start)):
        delete_files_in_range(folder, start[idx], end[idx], False)
