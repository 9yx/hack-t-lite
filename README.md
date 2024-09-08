# hack-t-lite

## Тестовый сервис:
Находится по адресу http://95.174.93.97:8081/assist

```bash
curl --request POST \
  --url 'http://95.174.93.97:8081/assist' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: insomnia/9.2.0' \
  --data '{
	"query": "You are a voice assistant. Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.\n\nGOALS:\n\n1. Хочу послать твит с поздравлением.\n\nConstraints:\n1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.\n2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.\n3. No user assistance\n4. Exclusively use the commands listed in double quotes e.g. \"command name\"\n5. Use subprocesses for commands that will not terminate within a few minutes\n\nCommands:\n1. Google Search: \"google\", args: \"input\": \"<search>\"\n2. Browse Website: \"browse_website\", args: \"url\": \"<url>\", \"question\": \"<what_you_want_to_find_on_website>\"\n3. Start GPT Agent: \"start_agent\", args: \"name\": \"<name>\", \"task\": \"<short_task_desc>\", \"prompt\": \"<prompt>\"\n4. Message GPT Agent: \"message_agent\", args: \"key\": \"<key>\", \"message\": \"<message>\"\n5. List GPT Agents: \"list_agents\", args:\n6. Delete GPT Agent: \"delete_agent\", args: \"key\": \"<key>\"\n7. Clone Repository: \"clone_repository\", args: \"repository_url\": \"<url>\", \"clone_path\": \"<directory>\"\n8. Write to file: \"write_to_file\", args: \"file\": \"<file>\", \"text\": \"<text>\"\n9. Read file: \"read_file\", args: \"file\": \"<file>\"\n10. Append to file: \"append_to_file\", args: \"file\": \"<file>\", \"text\": \"<text>\"\n11. Delete file: \"delete_file\", args: \"file\": \"<file>\"\n12. Search Files: \"search_files\", args: \"directory\": \"<directory>\"\n13. Analyze Code: \"analyze_code\", args: \"code\": \"<full_code_string>\"\n14. Get Improved Code: \"improve_code\", args: \"suggestions\": \"<list_of_suggestions>\", \"code\": \"<full_code_string>\"\n15. Write Tests: \"write_tests\", args: \"code\": \"<full_code_string>\", \"focus\": \"<list_of_focus_areas>\"\n16. Execute Python File: \"execute_python_file\", args: \"file\": \"<file>\"\n17. Generate Image: \"generate_image\", args: \"prompt\": \"<prompt>\"\n18. Send Tweet: \"send_tweet\", args: \"text\": \"<text>\"\n19. Do Nothing: \"do_nothing\", args:\n20. Task Complete (Shutdown): \"task_complete\", args: \"reason\": \"<reason>\"\n21. Analyze Expenses: \"analyze_expenses\", args: \"transactions\": \"<transaction_list>\"\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\nYou should only respond in JSON format as described below\nResponse Format:\n{\n    \"thoughts\": {\n        \"text\": \"thought\",\n        \"reasoning\": \"reasoning\",\n        \"plan\": \"- short bulleted\\n- list that conveys\\n- long-term plan\",\n        \"criticism\": \"constructive self-criticism\",\n        \"speak\": \"thoughts summary to say to user\"\n    },\n    \"command\": {\n        \"name\": \"command name\",\n        \"args\": {\n            \"arg name\": \"value\"\n        }\n    }\n}\nEnsure the response can be parsed by Python json.loads"
}'
```

В ответе:
```json
{
	"text": "{\"thoughts\": {\"text\": \"Поздравление с Днем рождения\", \"reasoning\": \"Пользователь хочет отправить твит с поздравлением, поэтому я начну с поиска подходящего поздравления и имени пользователя.\", \"plan\": [\"1. Поиск поздравлений в интернете\", \"2. Определение имени пользователя\", \"3. Формирование твитта\", \"4. Отправка твитта\"], \"criticism\": \"Необходимо проверить, что имя пользователя указано правильно и что поздравление не содержит ошибок.\", \"speak\": \"Сейчас я найду подходящее поздравление и подготовлю твит.\"}, \"command\": {\"name\": \"google\", \"args\": {\"input\": \"поздравления с днем рождения\"}}}"
}
```

## Описание файлов в проекте

- `prepare_dataset.ipynb` - ноутбук для предподготовки данных для датасета
- `create_dataset.ipynb` - ноутбук для создания датасета, который использовался для обучения модели https://huggingface.co/AnatoliiPotapov/T-lite-instruct-0.1, используя https://github.com/turbo-llm/turbo-alignment
- `dataset_analysis.ipynb` - анализ данных в созданном датасете
- `teach_model.ipynb` - ноутбук для вызова turbo-alignment для обучения адаптера модели
- `test_model.ipynb` - ноутбук для тестирования обученного адаптера 
- `main.py` и `models.py` - реалиазация openapi из описания задачи для проверки модели.
- `T-Lite-Project/T-Lite-Project` - подпроект с реализацией Telegram бота.
- `commands` - директория с JSON c списком команд у Ассистентов
- `goals` - директория с JSON c списком целей у Ассистентов
- `promts` - директория с JSON c списком промтов для оркестрации запросов.
- `datasets` - директория с выборками данных.

`config/t_bank_train.json` - конфиг используемый в обучении в `turbo-llm/turbo-alignment`:

```json
{
  "train_dataset_settings": {
    "sources": [
      {
        "name": "train_chat",
        "records_path": "./datasets/dataset-train.jsonl",
        "sample_rate": 1.0
      }
    ],
    "prompt_template": {
      "role_tag_mapping": {
        "bot": "assistant",
        "user": "user",
        "system": "system"
      },
      "prefix_template": "<|start_header_id|>{role}<|end_header_id|>\n\n",
      "suffix_template": "<|eot_id|>"
    },
    "dataset_type": "chat",
    "max_tokens_count": 2000,
    "only_answer_loss": true
  },
  "val_dataset_settings": {
    "sources": [
      {
        "name": "val_chat",
        "records_path": "./datasets/dataset-valid.jsonl",
        "sample_rate": 1.0
      }
    ],
    "prompt_template": {
      "role_tag_mapping": {
        "bot": "assistant",
        "user": "user",
        "system": "system"
      },
      "prefix_template": "<|start_header_id|>{role}<|end_header_id|>\n\n",
      "suffix_template": "<|eot_id|>"
    },
    "dataset_type": "chat",
    "max_tokens_count": 2000,
    "only_answer_loss": true
  },
  "model_settings": {
    "model_path": "./model/T-lite-instruct-0.1",
    "model_type": "causal",
    "transformers_settings": {
       "low_cpu_mem_usage":  true
    },
    "peft_settings": {
      "r": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
      ],
      "task_type": "CAUSAL_LM",
      "name": "LORA"
    },
    "model_kwargs": {
      "torch_dtype": "auto",
      "attn_implementation": "eager",
      "device_map": "auto"
    }
  },
  "cherry_pick_settings": {
    "generator_transformers_settings": {
      "num_beams": 3,
      "max_new_tokens": 512,
      "repetition_penalty": 1.02
    },
    "custom_generation_settings": {
      "skip_special_tokens": false
    },
    "dataset_settings": {
      "sources": [
        {
          "name": "cherrypick_chat",
          "records_path": "./datasets/dataset-train.jsonl",
          "sample_rate": 1.0
        }
      ],
      "prompt_template": {
        "role_tag_mapping": {
          "bot": "assistant",
          "user": "user",
          "system": "system"
        },
        "prefix_template": "<|start_header_id|>{role}<|end_header_id|>\n\n",
        "suffix_template": "<|eot_id|>"
      },
      "dataset_type": "chat",
      "max_tokens_count": 2000,
      "random_cut": true,
      "only_answer_loss": true
    },
    "metric_settings": [
      {
        "type": "length",
        "parameters": {
          "need_average": [true]
        }
      }
    ]
  },
  "tokenizer_settings": {},
  "trainer_settings": {
    "evaluation_strategy": "steps",
    "save_total_limit": 3,
    "load_best_model_at_end": true,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "logging_steps": 1,
    "eval_steps": 500,
    "save_steps": 500,
    "learning_rate": 1e-6,
    "num_train_epochs": 2,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.03,
    "fp16": true,
    "bf16": false,
    "optim": "adamw_8bit",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-6,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
  },
  "wandb_settings": {
    "project_name": "alignment",
    "run_name": "sft",
    "entity": "turbo-alignment",
    "mode": "disabled"
  },
  "seed": 0,
  "log_path": "./model/T-lite-instruct-adapter"
}
```
