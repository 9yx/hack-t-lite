import os
from fastapi import FastAPI, Body
import telebot, threading
import client
import uvicorn

BOT_TOKEN = os.environ["BOT_TOKEN"]

bot = telebot.TeleBot(BOT_TOKEN)
app = FastAPI()

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Привет")

@bot.message_handler(content_types=['text'])
def handle_message(message):
    response = client.send_dialog(message.chat.id, message.text)

    if response.get("command") == 'WAIT':
        bot.send_message(message.chat.id, "Дождись ответа =)")

@app.post("/send_answer/{id}")
async def start_audit(id,  data = Body()):
    bot.send_message(id, data["txt"])

def run_start_process():
    bot.polling(none_stop=True)

if __name__ == '__main__':
    thread1 = threading.Thread(target=run_start_process)
    thread1.start()

    uvicorn.run('main:app', workers=2, host="0.0.0.0", port=9001)



