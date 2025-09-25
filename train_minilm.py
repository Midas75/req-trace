from sentence_transformers import (
    SentenceTransformer,
    evaluation,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from conf_minilm import (
    batch_size,
    model_name,
    epochs,
    warmup_steps,
    learning_rate,
    gradient_accumulation_steps,
    half,
    use_gtk_cosine,
    count_topk_idx,
    optim
)
from dataset import from_pickle, to_dataset
from gtk_loss import (
    GroupTopkCosineSimilarityLoss,
    GroupTopkCosineSimilarityEvaluator,
    CosineSimilarityLossFP16,
)

model = SentenceTransformer(
    model_name,
    model_kwargs={
        # "attn_implementation": "flash_attention_2",
        "dtype": "half" if half else None,
        # "device_map": "auto",
    },
    tokenizer_kwargs={"padding_side": "left"},
)
model.gradient_checkpointing_enable()
train_examples, valid_tuples = from_pickle()
train_examples = to_dataset(train_examples)

if use_gtk_cosine:
    train_loss = GroupTopkCosineSimilarityLoss(model, count_idx=count_topk_idx)
    valid_evaluator = GroupTopkCosineSimilarityEvaluator(
        valid_tuples[0],
        valid_tuples[1],
        valid_tuples[2],
        show_progress_bar=True,
        batch_size=batch_size,
        group_size=train_loss.group_size,
        top_k=train_loss.top_k,
    )
else:
    train_loss = CosineSimilarityLossFP16(model)
    valid_evaluator = evaluation.EmbeddingSimilarityEvaluator(
        valid_tuples[0],
        valid_tuples[1],
        valid_tuples[2],
        show_progress_bar=True,
        batch_size=batch_size,
    )
train_args = SentenceTransformerTrainingArguments(
    output_dir="./trained_minilm" + ("_gtcs" if use_gtk_cosine else ""),
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    eval_steps=1,
    logging_steps=1,
    learning_rate=learning_rate,
    optim=optim,
    # fp16=True,
    report_to="none",
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_strategy="epoch",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_examples,
    loss=train_loss,
    evaluator=valid_evaluator,
)
trainer.train()

print(valid_evaluator(model))
if count_topk_idx:
    for k, v in train_loss.idx_counter.items():
        print(k, v)
