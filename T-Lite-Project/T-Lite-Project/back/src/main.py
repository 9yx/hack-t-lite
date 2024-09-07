from fastapi import FastAPI, Body
import uvicorn
import schedule, threading, logging
from pydantic import BaseModel

from client import send_bot
from db import mark_done, init_db, get_process_list, get_wait, create
from service import run

init_db()
app = FastAPI()

class Item(BaseModel):
    txt: str

if __name__ == '__main__':
    uvicorn.run('main:app', workers=1, host="0.0.0.0", port=9000)

@app.post("/send_dailog/{id}")
async def start_audit(id,  data = Body()):
    if get_wait(id) is not None:
        return {"command": "WAIT"}
    create(id, data["txt"])
    return {"command": "OK"}

def start_process():
    dialogs = get_process_list()
    for d in dialogs:
        try:
            answer = run(d.question)
            mark_done(d.id, answer)
            send_bot(d.user_id, answer)
        except Exception as e:
            logging.error(e)

scheduler1 = schedule.Scheduler()
scheduler1.every(1).seconds.do(start_process)
def run_start_process():
    while True:
        scheduler1.run_pending()
thread1 = threading.Thread(target=run_start_process)
thread1.start()


