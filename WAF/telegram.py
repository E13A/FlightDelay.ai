"""
Telegram Bot Integration Module
Sends detailed alert notifications to Telegram when security events occur.
"""

import requests
import datetime

# Telegram Bot Configuration
TELEGRAM_ENABLED = True  # Set True to enable Telegram notifications
TELEGRAM_BOT_TOKEN = "8399775048:AAGmmEfocaJaV5_qwhqCpSZ32H-5AqQwoZc"  # Your Telegram bot token (get from @BotFather)
TELEGRAM_CHAT_ID = "1469505998"  # Your Telegram chat ID (get from @userinfobot)
TELEGRAM_API_URL = "https://api.telegram.org/bot{}/sendMessage"


def escape_html(text):
    """Escape special characters for Telegram HTML."""
    if not text:
        return ""
    # Escape HTML special characters
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text

def send_alert(event_type, level, source_ip, message, details=None):
    """
    Send detailed alert message to Telegram bot.
    
    Args:
        event_type: Type of event (e.g., "SQLi", "XSS", "RateLimit")
        level: Severity level ("ERROR", "WARNING", "INFO")
        source_ip: Source IP address
        message: Alert message
        details: Optional detailed information string
    """
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        # Build detailed message with HTML formatting (more reliable than Markdown)
        emoji = "🔴" if level == "ERROR" else "🟡" if level == "WARNING" else "🔵"
        alert_msg = f"{emoji} <b>WAF Alert</b>\n\n"
        alert_msg += f"<b>Type:</b> {escape_html(event_type)}\n"
        alert_msg += f"<b>Level:</b> {escape_html(level)}\n"
        alert_msg += f"<b>Source IP:</b> <code>{escape_html(source_ip)}</code>\n"
        alert_msg += f"<b>Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        alert_msg += f"<b>Message:</b> {escape_html(message)}\n"
        
        if details:
            # Escape details but preserve newlines
            escaped_details = escape_html(details)
            alert_msg += f"\n<b>Details:</b>\n<pre>{escaped_details}</pre>"
        
        # Send to Telegram
        url = TELEGRAM_API_URL.format(TELEGRAM_BOT_TOKEN)
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": alert_msg,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        print(f"[Telegram] ✓ Alert sent: {event_type}")
    except requests.exceptions.HTTPError as e:
        # Try with plain text if HTML fails
        try:
            print(f"[Telegram] HTML failed, trying plain text: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[Telegram] Error response: {e.response.text}")
            
            emoji = "🔴" if level == "ERROR" else "🟡" if level == "WARNING" else "🔵"
            alert_msg = f"{emoji} WAF Alert\n\n"
            alert_msg += f"Type: {event_type}\n"
            alert_msg += f"Level: {level}\n"
            alert_msg += f"Source IP: {source_ip}\n"
            alert_msg += f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            alert_msg += f"Message: {message}\n"
            
            if details:
                alert_msg += f"\nDetails:\n{details}"
            
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": alert_msg
            }
            
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            print(f"[Telegram] ✓ Alert sent (plain text): {event_type}")
        except Exception as e2:
            print(f"[Telegram] ✗ Failed to send alert (both HTML and plain text failed): {e2}")
            if hasattr(e2, 'response') and e2.response is not None:
                print(f"[Telegram] Error response: {e2.response.text}")
    except Exception as e:
        print(f"[Telegram] ✗ Failed to send alert: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[Telegram] Response: {e.response.text}")

