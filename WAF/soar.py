"""
SOAR (Security Orchestration, Automation and Response) Module
Handles temporary IP banning for attack attempts (180 seconds).
"""

import time
from collections import defaultdict

# SOAR: Banned IPs (temporary bans from attack attempts - 180 seconds)
BAN_DURATION = 180  # Ban duration in seconds
banned_ips = defaultdict(float)  # IP -> unban timestamp


def ban_ip(ip: str):
    """Temporarily ban an IP address for 180 seconds (SOAR action)."""
    current_time = time.time()
    unban_time = current_time + BAN_DURATION
    banned_ips[ip] = unban_time
    print(f"[SOAR] IP {ip} has been temporarily banned for {BAN_DURATION} seconds due to attack attempt")


def unban_ip(ip: str) -> bool:
    """Manually remove ban from an IP address."""
    if ip in banned_ips:
        del banned_ips[ip]
        print(f"[SOAR] IP {ip} has been manually unbanned")
        return True
    return False


def is_banned(ip: str) -> bool:
    """Check if an IP is currently banned (and clean expired bans)."""
    current_time = time.time()
    
    # Clean expired bans
    expired_ips = [ip_addr for ip_addr, unban_time in banned_ips.items() if unban_time <= current_time]
    for expired_ip in expired_ips:
        del banned_ips[expired_ip]
        print(f"[SOAR] IP {expired_ip} ban expired automatically")
    
    # Check if IP is still banned
    if ip in banned_ips:
        return True
    return False


def get_banned_ips():
    """Get list of currently banned IPs with remaining time until unblock."""
    current_time = time.time()
    banned_list = []
    
    # Clean expired bans first
    expired_ips = [ip_addr for ip_addr, unban_time in banned_ips.items() if unban_time <= current_time]
    for expired_ip in expired_ips:
        del banned_ips[expired_ip]
    
    # Get active bans with remaining time
    for ip, unban_time in banned_ips.items():
        remaining_seconds = max(0, int(unban_time - current_time))
        banned_list.append({
            "ip": ip,
            "remaining_seconds": remaining_seconds,
            "unban_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unban_time))
        })
    
    return banned_list

