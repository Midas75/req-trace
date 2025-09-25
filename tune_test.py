from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import bitsandbytes as bnb

# 1. 初始化模型
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        # "attn_implementation": "flash_attention_2",
        # "dtype": "float16",
    },
    tokenizer_kwargs={"padding_side": "left"},
)
# 2. 构建训练数据
train_examples = [
    InputExample(texts=["What is AI?", "Artificial intelligence is AI."], label=1.0),
    InputExample(
        texts=["The capital of China?", "Beijing is the capital of China."], label=1.0
    ),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)

# 3. 选择 loss
train_loss = losses.CosineSimilarityLoss(model)

# 4. 验证训练前效果
sentences = [
    "What is AI?",
    "Artificial intelligence is AI.",
    "The capital of China?",
    "Beijing is the capital of China.",
]
embeddings_before = model.encode(sentences, convert_to_tensor=True)

cos_sim_before = util.cos_sim(embeddings_before[0], embeddings_before[1]).item()
print(f"Cosine similarity before training (AI pair): {cos_sim_before:.4f}")

cos_sim_before2 = util.cos_sim(embeddings_before[2], embeddings_before[3]).item()
print(f"Cosine similarity before training (China pair): {cos_sim_before2:.4f}")

# 5. fine-tuning
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    optimizer_class=bnb.optim.AdamW8bit,
)

# 6. 验证训练后效果
embeddings_after = model.encode(sentences, convert_to_tensor=True)

cos_sim_after = util.cos_sim(embeddings_after[0], embeddings_after[1]).item()
print(f"Cosine similarity after training (AI pair): {cos_sim_after:.4f}")

cos_sim_after2 = util.cos_sim(embeddings_after[2], embeddings_after[3]).item()
print(f"Cosine similarity after training (China pair): {cos_sim_after2:.4f}")

# 7. 保存模型
save_path = "./fine_tuned_qwen_embedding"
model.save(save_path)
print(f"Model saved to {save_path}")
