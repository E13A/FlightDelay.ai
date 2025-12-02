import re
import datetime
import time
import requests
import urllib.parse
from netfilterqueue import NetfilterQueue
from scapy.all import IP, TCP, Raw, send
import sys
from collections import defaultdict
import soar
import telegram
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

# Configuration
SIEM_INGEST_URL = "http://127.0.0.1:8000/ingest"  # Your API
NFQ_NUM = 1
BLOCKING_ENABLED = True  # Set True to drop matching packets
RATE_LIMIT_ENABLED = True  # Set True to enable rate limiting
RATE_LIMIT_REQUESTS = 100  # Max requests per time window
RATE_LIMIT_WINDOW = 60  # Time window in seconds

WARNING_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Request Blocked</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#111; color:#f8f8f8; text-align:center; padding-top:15vh; }}
    h1 {{ color:#ff6666; }}
    p {{ max-width:520px; margin:1rem auto; line-height:1.4; }}
    a {{ color:#66ccff; }}
  </style>
</head>
<body>
  <h1>Request Blocked</h1>
  <p>Your request triggered the Web Application Firewall and has been blocked.
     If you believe this is a mistake, please contact the administrator.</p>
</body>
</html>
"""

# Rate limiting: track requests per IP
rate_limit_tracker = defaultdict(list)  # IP -> list of timestamps

# Detection patterns (tuned conservative)
SQLI_PATTERNS = [
    re.compile(r"(?i)\bunion\b.*\bselect\b"),
    re.compile(r"(?i)\bselect\b.*\bfrom\b"),
    re.compile(r"(?i)\bor\s+1\s*=\s*1\b"),
    re.compile(r"(?i)\bbenchmark\(|sleep\(|pg_sleep\("),
    re.compile(r"(?i)--\s|/\*.*\*/"),  # comment injection markers
]
XSS_PATTERNS = [
    re.compile(r"(?i)<script\b"),
    re.compile(r"(?i)on\w+\s*="),
    re.compile(r"(?i)javascript:"),
    re.compile(r"(?i)<img\b.*onerror\b"),
]
# Command Injection patterns
CMD_INJECTION_PATTERNS = [
    re.compile(r"[;&|`]\s*(?:ls|cat|whoami|id|pwd|uname|ps|netstat)"),
    re.compile(r"(?i)\$\{.*\}|\$\(.*\)"),  # command substitution
    re.compile(r"[;&|`]\s*(?:rm|del|mkdir|echo)\s"),
    re.compile(r"(?i)(?:exec|system|passthru|shell_exec|eval)\s*\("),
]
# Path Traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./"),
    re.compile(r"\.\.\\\\"),
    re.compile(r"(?i)(?:\.\.%2f|\.\.%5c)"),  # URL encoded
    re.compile(r"(?i)(?:etc/passwd|boot\.ini|win\.ini)"),
]

# NoSQL Injection patterns
NOSQL_INJECTION_PATTERNS = [
    re.compile(r"(?i)\$ne|\$gt|\$lt|\$regex"),
    re.compile(r"(?i)\{\s*\$where"),
    re.compile(r"(?i)\{\s*\$ne\s*:"),
]

# Helper: send event to SIEM in dashboard format
def send_siem_event(source_ip, event_type, level, message, details=None):
    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "source": source_ip,
        "event_type": event_type,
        "level": level,
        "message": message
    }
    print(f"[WAF] Attempting to send alert: {event_type} from {source_ip} to {SIEM_INGEST_URL}")
    
    # Retry logic similar to log ingester
    for attempt in range(3):
        try:
            response = requests.post(SIEM_INGEST_URL, json=payload, timeout=2)
            response.raise_for_status()
            print(f"[WAF] ✓ Alert sent to SIEM: {event_type} from {source_ip} (Status: {response.status_code})")
            return
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                print(f"[WAF] Retry {attempt+1}/3: Failed to send alert to SIEM: {e}")
                time.sleep(0.5)
            else:
                print(f"[WAF] ✗ Failed to send alert to SIEM after 3 attempts: {e}")
                print(f"[WAF] Payload was: {payload}")
        except Exception as e:
            print(f"[WAF] ✗ Unexpected error sending alert: {e}")
            print(f"[WAF] Payload was: {payload}")

def send_warning_page(ip_layer, tcp_layer):
    """Send an HTTP warning page back to the client while blocking the request."""
    if not tcp_layer:
        return
    try:
        body = WARNING_PAGE_HTML.encode("utf-8")
        # Return a forbidden page directly (no auto-refresh)
        headers = (
            "HTTP/1.1 403 Forbidden\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")
        payload = headers + body

        # Determine payload length to advance ACK correctly
        try:
            payload_len = len(bytes(tcp_layer.payload))
        except Exception:
            payload_len = 0

        response_packet = (
            IP(src=ip_layer.dst, dst=ip_layer.src)
            / TCP(
                sport=tcp_layer.dport,
                dport=tcp_layer.sport,
                seq=tcp_layer.ack,
                ack=tcp_layer.seq + payload_len,
                flags="PA",
            )
            / Raw(load=payload)
        )
        send(response_packet, verbose=0)
        print(f"[WAF] Warning page sent to {ip_layer.src}")
    except Exception as exc:
        print(f"[WAF] Failed to send warning page: {exc}")

def send_rate_limit_response(ip_layer, tcp_layer, retry_after=60):
    """Send an HTTP 429 Too Many Requests response to the client while blocking the request."""
    if not tcp_layer:
        return
    try:
        body = b"429 Too Many Requests"
        headers = (
            "HTTP/1.1 429 Too Many Requests\r\n"
            "Content-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Retry-After: {retry_after}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")
        payload = headers + body

        # Determine payload length to advance ACK correctly
        try:
            payload_len = len(bytes(tcp_layer.payload))
        except Exception:
            payload_len = 0

        response_packet = (
            IP(src=ip_layer.dst, dst=ip_layer.src)
            / TCP(
                sport=tcp_layer.dport,
                dport=tcp_layer.sport,
                seq=tcp_layer.ack,
                ack=tcp_layer.seq + payload_len,
                flags="PA",
            )
            / Raw(load=payload)
        )
        send(response_packet, verbose=0)
        print(f"[WAF] Rate limit 429 sent to {ip_layer.src} (Retry-After: {retry_after}s)")
    except Exception as exc:
        print(f"[WAF] Failed to send rate limit response: {exc}")

def check_rate_limit(src_ip: str) -> bool:
    """
    Check if IP has exceeded rate limit.
    Returns True if rate limit exceeded, False otherwise.
    """
    if not RATE_LIMIT_ENABLED:
        return False
    
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    
    # Clean old entries for this IP
    rate_limit_tracker[src_ip] = [
        ts for ts in rate_limit_tracker[src_ip] if ts > window_start
    ]
    
    # Check if limit exceeded (before adding current request)
    if len(rate_limit_tracker[src_ip]) >= RATE_LIMIT_REQUESTS:
        return True
    
    # Add current request timestamp
    rate_limit_tracker[src_ip].append(current_time)
    return False

def get_rate_limited_ips():
    """
    Get list of currently rate limited IPs with their remaining time until unblock.
    Returns list of dicts: [{"ip": "1.2.3.4", "remaining_seconds": 45, "request_count": 100}, ...]
    """
    if not RATE_LIMIT_ENABLED:
        return []
    
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    rate_limited = []
    
    for ip, timestamps in rate_limit_tracker.items():
        # Skip internal tracking keys and banned IPs
        if ip.startswith("_") or soar.is_banned(ip):
            continue
            
        # Clean old entries
        valid_timestamps = [ts for ts in timestamps if ts > window_start]
        rate_limit_tracker[ip] = valid_timestamps
        
        # Check if currently rate limited
        if len(valid_timestamps) >= RATE_LIMIT_REQUESTS:
            # Calculate remaining time until oldest entry expires
            oldest_timestamp = min(valid_timestamps)
            unblock_time = oldest_timestamp + RATE_LIMIT_WINDOW
            remaining_seconds = max(0, int(unblock_time - current_time))
            
            rate_limited.append({
                "ip": ip,
                "remaining_seconds": remaining_seconds,
                "request_count": len(valid_timestamps),
                "unblock_at": datetime.datetime.fromtimestamp(unblock_time).isoformat()
            })
    
    return rate_limited

def clear_rate_limit_for_ip(ip: str):
    """Manually clear rate limit for a specific IP."""
    if ip in rate_limit_tracker:
        del rate_limit_tracker[ip]
        print(f"[WAF] Rate limit cleared for IP: {ip}")
        return True
    return False

# Helper: conservative HTTP detection - extract http request text from packet payload
def extract_http_from_payload(raw_payload: bytes) -> str | None:
    """
    Try to find an HTTP request inside the raw TCP payload bytes.
    Return decoded string (headers+body) or None.
    """
    try:
        text = raw_payload.decode("utf-8", errors="ignore")
    except Exception:
        return None

    # Very simple check for request-line presence
    if text.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ")):
        return text

    # If packet contains data but not at start, search for request line inside
    for m in ( "GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS " ):
        idx = text.find("\r\n" + m)
        if idx != -1:
            return text[idx+2:]
    return None

# Helper: parse HTTP request text into method, path, headers, body
def parse_http(http_text: str):
    """
    Returns (method, path_with_query, headers_dict, body_bytes)
    If parsing fails, raises ValueError.
    """
    # Split headers/body
    split_idx = http_text.find("\r\n\r\n")
    if split_idx == -1:
        # Maybe only headers present (GET)
        header_block = http_text
        body = ""
    else:
        header_block = http_text[:split_idx]
        body = http_text[split_idx+4:]

    lines = header_block.split("\r\n")
    request_line = lines[0]
    parts = request_line.split()
    if len(parts) < 2:
        raise ValueError("Invalid request line")

    method = parts[0]
    path = parts[1]  # may include query
    headers = {}
    for h in lines[1:]:
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return method, path, headers, body

# Detection: check url/query, body, and headers (decoded)
def detect_attacks(path: str, body: str, headers_text: str = "") -> (str | None):
    """
    Return attack type ('SQLi', 'XSS', 'CmdInjection', 'PathTraversal', 'NoSQLInjection') or None
    We percent-decode url and body first to catch encoded payloads.
    """
    # Decode percent encodings
    try:
        decoded_path = urllib.parse.unquote_plus(path)
    except Exception:
        decoded_path = path

    try:
        decoded_body = urllib.parse.unquote_plus(body)
    except Exception:
        decoded_body = body

    try:
        decoded_headers = urllib.parse.unquote_plus(headers_text)
    except Exception:
        decoded_headers = headers_text

    # Combine all text for checking
    all_text = f"{decoded_path} {decoded_body} {decoded_headers}"

    # Check for each attack type (order matters - check more specific first)
    # 1. SQL Injection
    for p in SQLI_PATTERNS:
        if p.search(all_text):
            return "SQLi"
    
    # 2. XSS
    for p in XSS_PATTERNS:
        if p.search(all_text):
            return "XSS"
    
    # 3. Command Injection
    for p in CMD_INJECTION_PATTERNS:
        if p.search(all_text):
            return "CmdInjection"
    
    # 4. Path Traversal
    for p in PATH_TRAVERSAL_PATTERNS:
        if p.search(all_text):
            return "PathTraversal"
    
    # 5. NoSQL Injection
    for p in NOSQL_INJECTION_PATTERNS:
        if p.search(all_text):
            return "NoSQLInjection"

    return None

# Callback for netfilterqueue packets
_packet_counter = 0

def process_packet(pkt):
    global _packet_counter
    _packet_counter += 1
    if _packet_counter % 50 == 0:
        print(f"[WAF] Processed {_packet_counter} packets...")
    
    try:
        raw = pkt.get_payload()  # raw packet bytes (IP packet)
        # attempt to get source IP via scapy; fallback to metadata
        src_ip = None
        tcp_payload = None
        tcp_layer = None
        try:
            ip = IP(raw)
            src_ip = ip.src
            # Extract TCP payload if TCP layer exists
            if ip.haslayer(TCP):
                tcp_layer = ip[TCP]
            if ip.haslayer(TCP) and ip.haslayer(Raw):
                tcp_payload = ip[Raw].load
        except Exception as e:
            # leave src_ip None; we'll set to 'unknown' later
            print(f"[WAF] Error parsing IP packet: {e}")
            src_ip = "unknown"
            pkt.accept()
            return

        if not tcp_payload:
            pkt.accept()
            return

        http_text = extract_http_from_payload(tcp_payload)
        if not http_text:
            pkt.accept()
            return

        # Parse request
        try:
            method, path, headers, body = parse_http(http_text)
        except Exception as e:
            print(f"[WAF] Error parsing HTTP: {e}")
            pkt.accept()
            return

        # Only inspect typical web requests (Host header present)
        if "host" not in headers:
            pkt.accept()
            return

        # Check if IP is banned (SOAR - temporary ban from attack attempts)
        if soar.is_banned(src_ip or "unknown"):
            message = f"{src_ip} {method} {path} -> blocked (banned IP)"
            print(f"[WAF] BLOCKED BANNED IP: {method} {path} from {src_ip}")
            
            # Get ban details
            banned_list = soar.get_banned_ips()
            ban_info = next((b for b in banned_list if b["ip"] == src_ip), None)
            details = None
            if ban_info:
                minutes = ban_info["remaining_seconds"] // 60
                seconds = ban_info["remaining_seconds"] % 60
                time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                details = f"Ban expires in: {time_str}\nUnban at: {ban_info['unban_at']}"
            
            # Send SIEM event
            send_siem_event(src_ip or "unknown", "BannedIP", "ERROR", message, details)
            
            # Send 403 response and drop packet
            if tcp_layer:
                send_warning_page(ip, tcp_layer)
            print("[WAF] Dropping packet (IP is banned).")
            pkt.drop()
            return

        # Check rate limiting first
        if check_rate_limit(src_ip or "unknown"):
            message = f"{src_ip} {method} {path} -> rate limit exceeded"
            print(f"[WAF] RATE LIMIT EXCEEDED: {method} {path} from {src_ip}")
            
            # Get rate limit details
            rate_limited = get_rate_limited_ips()
            rate_info = next((r for r in rate_limited if r["ip"] == src_ip), None)
            details = None
            if rate_info:
                minutes = rate_info["remaining_seconds"] // 60
                seconds = rate_info["remaining_seconds"] % 60
                time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                details = f"Request count: {rate_info['request_count']}\nUnblock in: {time_str}\nUnblock at: {rate_info['unblock_at']}"
            
            # Send SIEM event
            send_siem_event(src_ip or "unknown", "RateLimit", "WARNING", message, details)
            
            # Send 429 response and drop packet
            if tcp_layer:
                send_rate_limit_response(ip, tcp_layer, RATE_LIMIT_WINDOW)
            print("[WAF] Dropping packet (rate limit exceeded).")
            pkt.drop()
            return

        # Detect attacks (path includes query, body, and headers like User-Agent)
        header_values = " ".join(headers.values()) if headers else ""
        attack = detect_attacks(path, body, header_values)

        if attack:
            # Build message
            message = f"{src_ip} {method} {path} -> detected {attack}"
            print(f"[WAF] DETECTED {attack}: {method} {path} from {src_ip}")

            # Build detailed information for Telegram
            details = f"*Attack Type:* {attack}\n"
            details += f"*Method:* {method}\n"
            details += f"*Path:* `{path}`\n"
            if body:
                body_preview = body[:100] + "..." if len(body) > 100 else body
                details += f"*Body:* `{body_preview}`\n"
            if header_values:
                headers_preview = header_values[:200] + "..." if len(header_values) > 200 else header_values
                details += f"*Headers:* `{headers_preview}`\n"
            details += f"*Action:* IP temporarily banned for 180 seconds"

            # Send ONE Telegram alert for this attack
            telegram.send_alert(attack, "ERROR", src_ip or "unknown", message, details)

            # SOAR: Temporarily ban IP on attack attempt (180 seconds)
            soar.ban_ip(src_ip or "unknown")
            send_siem_event(src_ip or "unknown", "SOAR_Banned", "ERROR", f"IP {src_ip} temporarily banned for 180s due to {attack} attack", details)

            # Send SIEM event
            send_siem_event(src_ip or "unknown", attack, "WARNING", message, details)

            if BLOCKING_ENABLED:
                # Log block event and drop packet
                send_siem_event(src_ip or "unknown", "Blocked Request", "ERROR", f"Dropped {method} {path} ({attack})", details)
                if tcp_layer:
                    send_warning_page(ip, tcp_layer)
                print("[WAF] Dropping packet (blocking enabled).")
                pkt.drop()
                return
            else:
                # monitoring only
                pkt.accept()
                return

        # No match — accept
        pkt.accept()

    except Exception as e:
        # On any unexpected error, accept the packet to avoid disrupting service.
        print(f"[WAF] Error processing packet: {e}")
        import traceback
        traceback.print_exc()
        try:
            pkt.accept()
        except Exception:
            pass

# HTTP server for rate limit and SOAR status API
class RateLimitHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/rate-limited":
            rate_limited = get_rate_limited_ips()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(rate_limited).encode())
        elif self.path == "/banned":
            banned = soar.get_banned_ips()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(banned).encode())
        elif self.path.startswith("/unban/"):
            try:
                import urllib.parse
                ip = urllib.parse.unquote(self.path.split("/unban/")[1])
                print(f"[WAF API] Received GET unban request for IP: {ip}")
                success = soar.unban_ip(ip)
                print(f"[WAF API] Unban result for {ip}: {success}")
                self.send_response(200 if success else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "ip": ip}).encode())
            except Exception as e:
                print(f"[WAF API] Error processing unban: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path.startswith("/unban/"):
            try:
                import urllib.parse
                ip = urllib.parse.unquote(self.path.split("/unban/")[1])
                print(f"[WAF API] Received unban request for IP: {ip}")
                success = soar.unban_ip(ip)
                print(f"[WAF API] Unban result for {ip}: {success}")
                self.send_response(200 if success else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": success, "ip": ip}).encode())
            except Exception as e:
                print(f"[WAF API] Error processing unban: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
        elif self.path.startswith("/clear-rate-limit/"):
            ip = self.path.split("/clear-rate-limit/")[1]
            success = clear_rate_limit_for_ip(ip)
            self.send_response(200 if success else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "ip": ip}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def start_rate_limit_api():
    """Start HTTP server for rate limit status API"""
    server = HTTPServer(("127.0.0.1", 8001), RateLimitHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[WAF] Rate limit API server started on http://127.0.0.1:8001/rate-limited")

# Entrypoint
def main():
    print("Starting NFQUEUE WAF (OWASP Top 10 detection: SQLi, XSS, CmdInjection, PathTraversal, NoSQLInjection).")
    print(f"SIEM endpoint: {SIEM_INGEST_URL}")
    print(f"Blocking mode: {'ON' if BLOCKING_ENABLED else 'OFF'}")
    print(f"Rate limiting: {'ON' if RATE_LIMIT_ENABLED else 'OFF'} ({RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s)")
    print(f"Telegram notifications: {'ON' if telegram.TELEGRAM_ENABLED and telegram.TELEGRAM_BOT_TOKEN and telegram.TELEGRAM_CHAT_ID else 'OFF'}")
    if telegram.TELEGRAM_ENABLED and (not telegram.TELEGRAM_BOT_TOKEN or not telegram.TELEGRAM_CHAT_ID):
        print("⚠️  Telegram bot token or chat ID not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in telegram.py to enable notifications.")
    
    # Start rate limit API server
    if RATE_LIMIT_ENABLED:
        start_rate_limit_api()
    
    nfq = NetfilterQueue()
    try:
        nfq.bind(NFQ_NUM, process_packet)
        print(f"Bound NFQUEUE {NFQ_NUM}. Waiting for packets...")
        nfq.run()
    except KeyboardInterrupt:
        print("Stopping WAF (KeyboardInterrupt).")
    except Exception as e:
        print("Error running NFQueue:", e)
    finally:
        try:
            nfq.unbind()
        except Exception:
            pass

if __name__ == "__main__":
    main()