#!/usr/bin/env python3
import time
import requests
from dateutil import parser

LOG_FILE = "/var/log/apache2/access.log"      # Path to Apache logs
SIEM_URL = "http://127.0.0.1:8000/ingest"    # SIEM endpoint
POST_TIMEOUT = 2                             # Seconds

# Convert status code into SIEM severity level
def status_to_level(status_code):
    try:
        code = int(status_code)
        if code >= 500:
            return "ERROR"
        elif code >= 400:
            return "WARNING"
        else:
            return "INFO"
    except:
        return "INFO"

# Parse a single Apache access log line
def parse_log_line(line):
    try:
        parts = line.split()
        ip = parts[0]
        
        # Normalize IPv6 loopback to IPv4 for consistency
        if ip == "::1":
            ip = "127.0.0.1"

        # Extract timestamp inside [ ... ]
        timestamp_str = line.split("[")[1].split("]")[0]
        timestamp = parser.parse(timestamp_str.replace(":", " ", 1))

        method = parts[5].strip('"')
        url = parts[6]
        status = parts[8]

        return {
            "source": ip,
            "event_type": url,                     # no detection — raw URL only
            "level": status_to_level(status),
            "message": line.strip(),              # raw full log line
            "timestamp": timestamp.isoformat()
        }

    except Exception:
        return None


# Reliable event sender with retry logic
def send_event(event):
    for attempt in range(3):
        try:
            response = requests.post(SIEM_URL, json=event, timeout=POST_TIMEOUT)
            response.raise_for_status()  # Check if request was successful
            print(f"Sent event: {event['event_type']} [{event['level']}]")
            return
        except Exception as e:
            print(f"[Retry {attempt+1}/3] Failed to send: {e}")
            time.sleep(1)


# Live "tail -f" for access.log
def tail_log(file_path):
    with open(file_path, "r") as f:
        f.seek(0, 2)  # Move to end of file

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue

            event = parse_log_line(line)
            if event:
                send_event(event)


if __name__ == "__main__":
    tail_log(LOG_FILE)
