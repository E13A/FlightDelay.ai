# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from app import db, schemas
from sqlalchemy.orm import Session
from sqlalchemy import or_
import datetime
from typing import List

from .db import SessionLocal, Event

app = FastAPI(title="SIEM-MVP")

# Initialize DB
db.init_db()

def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@app.get("/")
def root():
    return {"message": "SIEM MVP running. Use POST /ingest to send events."}

@app.post("/ingest", status_code=201)
def ingest_event(event: schemas.IngestEvent, session: Session = Depends(get_db)):
    try:
        # Handle timestamp - Pydantic should parse ISO strings to datetime, but handle edge cases
        if event.timestamp:
            ts = event.timestamp
        else:
            ts = datetime.datetime.utcnow()

        ev = Event(
            timestamp=ts,
            source=event.source,
            event_type=event.event_type,
            level=event.level,
            message=event.message
        )

        session.add(ev)
        session.commit()
        session.refresh(ev)

        return {"status": "ok", "id": ev.id}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error ingesting event: {str(e)}")

@app.get("/events", response_model=List[schemas.IngestEvent])
def get_events(session: Session = Depends(get_db)):
    """
    Return recent events plus the latest WAF alerts so the dashboard can
    always display security detections even if raw logs are noisy.
    """
    recent_events = session.query(Event).order_by(Event.timestamp.desc()).limit(100).all()

    waf_events = (
        session.query(Event)
        .filter(
            or_(
                Event.event_type.in_(["SQLi", "XSS", "Blocked Request"]),
                Event.message.ilike("%detected%"),
                Event.message.ilike("%Blocked Request%"),
            )
        )
        .order_by(Event.timestamp.desc())
        .limit(100)
        .all()
    )

    events_by_id = {}
    for ev in recent_events + waf_events:
        events_by_id[ev.id] = ev

    events = sorted(events_by_id.values(), key=lambda e: e.timestamp, reverse=True)

    return [
        {
            "timestamp": ev.timestamp,
            "source": ev.source,
            "event_type": ev.event_type,
            "level": ev.level,
            "message": ev.message,
        }
        for ev in events
    ]

@app.get("/rate-limited")
def get_rate_limited_ips():
    """
    Get list of currently rate limited IPs with remaining time until unblock.
    This queries the WAF's rate limit tracker via HTTP API.
    """
    try:
        import requests
        response = requests.get("http://127.0.0.1:8001/rate-limited", timeout=1)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        # WAF API not available or not running
        return []

@app.get("/banned")
def get_banned_ips():
    """
    Get list of permanently banned IPs (SOAR).
    This queries the WAF's banned IPs list via HTTP API.
    """
    try:
        import requests
        response = requests.get("http://127.0.0.1:8001/banned", timeout=1)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        # WAF API not available or not running
        return []

@app.get("/unban/{ip}")
def unban_ip_endpoint(ip: str):
    """
    Remove ban from an IP address (SOAR action).
    """
    try:
        import requests
        import urllib.parse
        # URL encode the IP in case it contains special characters
        encoded_ip = urllib.parse.quote(ip, safe='')
        response = requests.get(f"http://127.0.0.1:8001/unban/{encoded_ip}", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"success": False, "error": f"IP {ip} not found in banned list"}
        else:
            return {"success": False, "error": f"WAF API returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to WAF API. Make sure WAF is running on port 8001."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/clear-rate-limit/{ip}")
def clear_rate_limit_endpoint(ip: str):
    """
    Manually clear rate limit for a specific IP.
    """
    try:
        import requests
        response = requests.post(f"http://127.0.0.1:8001/clear-rate-limit/{ip}", timeout=1)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": "WAF API not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}
