from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ProcessedResume(Base):
    """Model for storing processed resumes and their predictions"""
    __tablename__ = 'processed_resumes'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    extracted_text = Column(Text)
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    skills = Column(Text)  # Stored as JSON
    education = Column(Text)  # Stored as JSON
    predicted_department = Column(String(100))
    predicted_sub_department = Column(String(100))
    department_confidence = Column(Float)
    sub_department_confidence = Column(Float)
    feedback_correct = Column(Integer, default=0)  # 1 for correct, -1 for incorrect
    created_at = Column(DateTime, default=datetime.utcnow)
    
class SkillKeyword(Base):
    """Model for storing and updating skill keywords"""
    __tablename__ = 'skill_keywords'
    
    id = Column(Integer, primary_key=True)
    skill_name = Column(String(100), unique=True)
    variations = Column(Text)  # Stored as JSON
    frequency = Column(Integer, default=1)
    last_seen = Column(DateTime, default=datetime.utcnow)

# Database setup function
def setup_database(db_url='sqlite:///resumes.db'):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
