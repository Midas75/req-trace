from conf_minilm import half, use_gtk_cosine, eval_topk
from sentence_transformers import (
    SentenceTransformer,
    evaluation,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import pairwise_cos_sim
from dataset import from_pickle
from gtk_loss import pairwise_gtk_cos_sim
from tqdm import tqdm
from torch import Tensor
import os


def clear_files_only(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.remove(os.path.join(root, f))


comp = pairwise_gtk_cos_sim if use_gtk_cosine else pairwise_cos_sim
model = SentenceTransformer(
    f"trained_minilm{'_gtcs' if use_gtk_cosine else ''}/checkpoint-2670",
    # f"trained_qwen3/checkpoint-1214",
    # "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        # "attn_implementation": "flash_attention_2",
        "dtype": "half" if half else None,
        # "device_map": "auto",
    },
    tokenizer_kwargs={"padding_side": "left"},
).eval()
train_examples, valid_tuples = from_pickle()
true_relevant = 0
false_relevant = 0
retrieved = 0
true_code = 0
eval_data = dict[str, set[str]]()
vembs = dict[str, Tensor]()
for v in tqdm(valid_tuples[1]):
    vembs[v] = model.encode(v, convert_to_numpy=True)
for i in range(len(valid_tuples[0])):
    req = valid_tuples[0][i]
    code = valid_tuples[1][i]
    label = valid_tuples[2][i]
    if req not in eval_data:
        eval_data[req] = set[str]()
    if label > 0.5:
        eval_data[req].add(code)
false_counter = 0
clear_files_only("false")
for k, vs in tqdm(eval_data.items()):
    true_code += len(vs)
    kemb = model.encode(k, convert_to_numpy=True)
    vcs = list[tuple[float, str]]()
    for codek, codeemb in vembs.items():
        c = comp([kemb], [codeemb])
        vcs.append((float(c), codek))
    sorted_vcs = sorted(vcs, key=lambda x: x[0], reverse=True)
    for i in range(0, min(eval_topk, len(vs))):
        retrieved += 1
        if sorted_vcs[i][1] in vs:
            true_relevant += 1
        else:
            if True:
                false_counter += 1
                f = open(f"false/false_{false_counter}.txt", "w")
                f.write(
                    "does the requirement have relation with the code? Answer me in Chinese\n\n"
                )
                f.write(str(sorted_vcs[i][0]))
                f.write("\n######################\n")
                f.write(sorted_vcs[i][1])
                f.write("\n######################\n" + k)
            false_relevant += 1
print(f"Precision: {true_relevant}/{retrieved} = {true_relevant/retrieved*100:.2f}%")
print(f"Recall: {true_relevant}/{true_code} = {true_relevant/true_code*100:.2f}%")
