import torch
import bitsandbytes
from tqdm import tqdm
from sentence_transformers.util import pairwise_cos_sim

from sentence_transformers import (
    SentenceTransformer,
)
from dataset import from_pickle, to_dataset
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
    dynamic_train,
    eval_topk,
)
from gtk_loss import (
    pairwise_gtk_cos_sim,
    GroupTopkCosineSimilarityLoss,
    CosineSimilarityLossFP16,
)
from torch import Tensor
from eval import do_eval

comp = pairwise_gtk_cos_sim if use_gtk_cosine else pairwise_cos_sim
model = SentenceTransformer(
    model_name,
    model_kwargs={"dtype": "half" if half else None},
    tokenizer_kwargs={"padding_side": "left"},
    device="cuda:0",
)
model.gradient_checkpointing_enable()
td, vd, ie = from_pickle()
if use_gtk_cosine:
    train_loss = GroupTopkCosineSimilarityLoss(model, count_idx=count_topk_idx)
    sim_func = pairwise_gtk_cos_sim
else:
    train_loss = CosineSimilarityLossFP16(model)
    sim_func = pairwise_cos_sim
if optim == "adamw_torch":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif optim == "adamw_bnb_8bit":
    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if not dynamic_train:
    train_dataloader = torch.utils.data.DataLoader(
        to_dataset(ie), shuffle=False, batch_size=batch_size
    )
    tokens = dict[int, tuple[list[dict[str, Tensor]], Tensor]]()
    for step, batch in tqdm(enumerate(train_dataloader), desc="tokenizing"):
        f1, f2 = model.tokenize(batch["text1"]), model.tokenize(batch["text2"])
        sf = [f1, f2]
        for f in sf:
            for tk, v in f.items():
                f[tk] = v.to(model.device)
        tokens[step] = (sf, batch["label"].detach().clone().to(model.device))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            train_dataloader, desc=f"training {epoch+1}/{epochs}", leave=False
        )

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            sf, labels = tokens[step]
            loss_value = train_loss(sf, labels)
            loss_value.backward()
            optimizer.step()
        thr, tkr, _ = do_eval(model, batch_size, sim_threshold, eval_topk, sim_func, vd)
        print(f"Epoch {epoch+1}/{epochs}")
        print(thr)
        print(tkr)
model.save(
    f"./trained_minilm_{'gtcs' if use_gtk_cosine else 'cs'}_{'dynamic' if dynamic_train else 'random'}"
)
thr, tkr, _ = do_eval(model, batch_size, sim_threshold, eval_topk, sim_func, vd)
print(thr)
print(tkr)
