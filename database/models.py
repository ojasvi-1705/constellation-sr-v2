"""SQLAlchemy models: InferenceRequest and Metric"""
import os
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    model = Column(String)
    filename = Column(String)
    status = Column(String, default="processing")  # processing | completed | failed
    input_size = Column(String)
    output_size = Column(String)
    processing_time = Column(Float)
    output_path = Column(String)
    error = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True)
    request_id = Column(String, ForeignKey("inference_requests.id"))
    psnr = Column(Float)
    ssim = Column(Float)
    sam = Column(Float)


url = os.getenv("DATABASE_URL", "sqlite:///./constellation.db")
engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)
