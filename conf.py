from typing import Literal

batch_size = 8
# model_name = "Qwen/Qwen3-Embedding-0.6B"
model_name = "google/embeddinggemma-300m"
epochs = 4
datasets_path = "data"
warmup_steps = 10
learning_rate = 2e-5
gradient_accumulation_steps = 1
openai_url = "https://ark.cn-beijing.volces.com/api/v3"
openai_model = "ep-20250914142034-fr87r"
openai_apikey = "76a878dc-4905-44c3-858c-8ef33006250f"
half = False
use_gtk_cosine = True
count_topk_idx = False
optim: Literal["adamw_torch", "adamw_bnb_8bit"] = "adamw_torch"
