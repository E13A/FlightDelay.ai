# app/db.py
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DATABASE_URL = os.getenv("SIEM_DATABASE_URL", "sqlite:///./siem.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    source = Column(String(128), nullable=True, index=True)   # e.g., service name or IP
    event_type = Column(String(512), nullable=True, index=True)
    level = Column(String(16), nullable=True, index=True)     # e.g., INFO/WARN/ERROR
    message = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

