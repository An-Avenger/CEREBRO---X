from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, index=True) # subject_id
    records = relationship("PredictionRecord", back_populates="patient", cascade="all, delete-orphan")

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Clinical Features of the visit
    age = Column(Float)
    educ = Column(Float)
    ses = Column(Float)
    mmse = Column(Float)
    cdr = Column(Float)
    nwbv = Column(Float)
    etiv = Column(Float)
    asf = Column(Float)
    
    # Genetics & Biomarkers
    apoe4 = Column(Integer, nullable=True) # 1 for True, 0 for False, Null for unknown
    p_tau = Column(Float, nullable=True) # pg/mL
    
    # Model Outputs
    predicted_cdr_class = Column(Integer)
    predicted_cdr_value = Column(Float)
    predicted_cdr_label = Column(String)
    class_probabilities = Column(JSON) # Store dict as JSON
    z_t_trajectory = Column(JSON, nullable=True) # The abstract brain state

    patient = relationship("Patient", back_populates="records")
