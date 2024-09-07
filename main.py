from __future__ import annotations
from typing import Union
from fastapi import FastAPI
from models import HTTPValidationError, Request, Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import re
import json


app = FastAPI(
    title='Assistant API',
    version='0.1.0',
)


torch.manual_seed(42)

#model_name = "/home/user1/environments/train_model/train_output_old/trainer/"
#model_name="/home/user1/environments/train_model/model/T-lite-instruct-0.1"
model_name="/home/user1/environments/train_model/model/T-lite-instruct-adapter/trainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.generation_config.pad_token_id = tokenizer.pad_token_id

def process_prompt(query):
    messages = [
        {"role": "system", "content": f"{query}"}, {"role": "user", "content": 
"""Ответь только JSON без другого текста. Ответ в строгом соответствии с форматом JSON.
Сторого соблюдай следующие моменты при составление ответа в формате JSON:
1. Используйте только двойные кавычки для строковых значений.
2. Не добавляйте запятые после последнего элемента в объектах или массивах.
3. Убедитесь, что все открывающие скобки имеют соответствующие закрывающие скобки.
4. Проверьте валидность JSON перед отправкой ответа.
Строго следуйте формату JSON при составлении вашего ответа.
Значение поля args в JSON заполняй из [Goals].
При заполнении поля args в JSON данные должны быть достоверны и не протиречивы, если данных нет, оставляй поле пустым.
 """}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("*****", generated_text, "*****")

    # Извлечение части ответа между тегами
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
    match = re.search(pattern, generated_text, re.DOTALL)

    if match:
       if match.group(1) is None:
          print("Error fix regexp")
          return ""
       assistant_response = match.group(1).strip()
       assistant_response = assistant_response.replace("```json", "").replace("```", "")
       try:
          json_data = json.loads(assistant_response)
          return json.dumps(json_data, ensure_ascii=False)
       except Exception as e:
          print("Error parse json", e)
          return ""
    else:
       print("Ответ ассистента не найден.")
       return ""

@app.post(
    '/assist',
    response_model=Response,
    responses={'422': {'model': HTTPValidationError}},
    tags=['default'],
)
async def assist_assist_post(body: Request) -> Union[Response, HTTPValidationError]:
    """
    Assist
    """
    response = process_prompt(body.query)
    return Response(text=response)

if __name__ == '__main__':
    uvicorn.run('main:app', workers=1, host="0.0.0.0", port=8081)

