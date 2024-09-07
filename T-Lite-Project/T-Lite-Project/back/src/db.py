import os

from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, select
from sqlalchemy import create_engine, Column, Integer, String, ARRAY, DateTime
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = os.environ["DATABASE_URL"]
Base = declarative_base()

class Dialog(Base):
    __tablename__ = 'Dialog'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=False)
    question = Column(String, unique=False)
    answer = Column(String, unique=False)
    status = Column(String, unique=False)


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def create(user_id, answer):
    with Session(engine) as session:
        dialog = Dialog(user_id=user_id, question =answer ,status="WAIT")
        session.add(dialog)
        session.commit()
        session.refresh(dialog)
        id = dialog.id
    return id

def get_wait(user_id):
    with Session(engine) as session:
        statement = select(Dialog).where(Dialog.status == "WAIT" and Dialog.user_id == user_id)
        questions = session.exec(statement)
        result = questions.first()
    return result

def get_process_list():
    with Session(engine) as session:
        statement = select(Dialog).where(Dialog.status == "WAIT").order_by(Dialog.id).limit(1)
        questions = session.exec(statement)
        result = questions.all()
    return result

def mark_done(id, answer):
    with Session(engine) as session:
        statement = select(Dialog).where(Dialog.id == id)
        results = session.exec(statement)
        d = results.one()
        d.status = "DONE"
        d.answer = answer

        session.add(d)
        session.commit()
        session.refresh(d)

        session.add(d)
        session.commit()
        session.refresh(d)