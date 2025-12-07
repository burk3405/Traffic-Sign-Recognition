import csv
import os

log_path = "logs/detection_log.csv"

if not os.path.exists(log_path):
    print("No log found.")
    exit()

fp = 0
total = 0

with open(log_path, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        img, count = row
        count = int(count)
        total += 1
        if count > 1:   # >1 candidates considered false positive case
            fp += (count - 1)

print("Total images:", total)
print("False positive detections:", fp)
print("FP per image:", fp / total if total > 0 else 0)
