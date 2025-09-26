from conf_minilm import (
    use_gtk_cosine,
    batch_size,
    model_name,
    epochs,
    learning_rate,
    half,
    count_topk_idx,
    optim,
    sim_threshold,
)
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
)
from sentence_transformers.util import pairwise_cos_sim
import torch
import bitsandbytes
from tqdm import tqdm
from dataset import to_req_code_matrix, from_pickle, to_dataset

from gtk_loss import (
    GroupTopkCosineSimilarityEvaluator,
    GroupTopkCosineSimilarityLoss,
    CosineSimilarityLossFP16,
    pairwise_gtk_cos_sim,
)

comp = pairwise_gtk_cos_sim if use_gtk_cosine else pairwise_cos_sim
model = SentenceTransformer(
    model_name,
    model_kwargs={"dtype": "half" if half else None},
    tokenizer_kwargs={"padding_side": "left"},
)
max_neg_ratio: float = 1
model.gradient_checkpointing_enable()
train_examples, valid_tuples = from_pickle()
rc_matrix = to_req_code_matrix(train_examples)

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

if optim == "adamw_torch":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif optim == "adamw_bnb_8bit":
    optimizer = bitsandbytes.adam.Adam8bit(model.parameters(), lr=learning_rate)

true_data = list[InputExample]()
contents = set[str]()
for req, codes in rc_matrix.items():
    contents.add(req)
    for code, label in codes.items():
        contents.add(code)
        if label < sim_threshold:
            continue
        true_data.append(InputExample(texts=[req, code], label=label))
last_error_data = list[InputExample]()
for epoch in range(epochs):
    model.eval()
    all_embs = dict[str, torch.Tensor]()
    last_error_data.clear()
    for content in tqdm(
        list(contents), desc=f"encoding {epoch+1}/{epochs}", leave=True
    ):
        all_embs[content] = model.encode(req)
    for req, codes in tqdm(
        rc_matrix.items(), desc=f"evaluating {epoch+1}/{epochs}", leave=True
    ):
        false_code = list[tuple[float, str]]()
        true_code_counter = 0
        for code, label in codes.items():
            if label > sim_threshold:
                true_code_counter += 1
            sim_score = float(comp([all_embs[req]], [all_embs[code]]))
            if sim_score > sim_threshold and label < sim_threshold:
                false_code.append((sim_score, code))
        false_code = sorted(false_code, key=lambda x: x[0], reverse=True)
        for score, code in false_code[: int(true_code_counter * max_neg_ratio)]:
            last_error_data.append(
                InputExample(texts=[req, code], label=rc_matrix[req][code])
            )
    print(f"error link count: {len(last_error_data)}")
    model.train()
    total_loss = 0
    train_dataloader = torch.utils.data.DataLoader(
        to_dataset(true_data + last_error_data), batch_size=batch_size
    )

    progress_bar = tqdm(
        train_dataloader, desc=f"training {epoch+1}/{epochs}", leave=True
    )

    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        f1, f2 = model.tokenize(batch["text1"]), model.tokenize(batch["text2"])
        sf = [f1, f2]
        for k in range(len(sf)):
            for tk, v in sf[k].items():
                if isinstance(v, torch.Tensor):
                    sf[k][tk] = v.to(model.device)
        labels = batch["label"].detach().clone().to(model.device)
        loss_value = train_loss(sf, labels)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
        # progress_bar.set_postfix(step=step)
    print(
        f"Epoch {epoch+1}/{epochs} - Avg loss: {total_loss/len(train_dataloader):.4f}"
    )

model.save("./trained_dynamic_minilm" + ("_gtcs" if use_gtk_cosine else ""))
print(valid_evaluator(model))
