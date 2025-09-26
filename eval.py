from conf_minilm import half, use_gtk_cosine, sim_threshold
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.util import pairwise_cos_sim
from dataset import from_pickle
from gtk_loss import pairwise_gtk_cos_sim
from tqdm import tqdm
from torch import Tensor
import os

sim_threshold = sim_threshold


def clear_files_only(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.remove(os.path.join(root, f))


comp = pairwise_gtk_cos_sim if use_gtk_cosine else pairwise_cos_sim
model = SentenceTransformer(
    f"trained_dynamic_minilm{'_gtcs' if use_gtk_cosine else ''}",
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
clear_files_only("false_case")
for k, vs in tqdm(eval_data.items()):
    true_code += len(vs)
    kemb = model.encode(k, convert_to_numpy=True)
    vcs = list[tuple[float, str]]()
    for codek, codeemb in vembs.items():
        c = float(comp([kemb], [codeemb]))
        need_save = False
        if c > sim_threshold and codek in vs:
            true_relevant += 1
            retrieved += 1
        elif c > sim_threshold and codek not in vs:
            false_relevant += 1
            false_counter += 1
            retrieved += 1
            need_save = True
        elif c < sim_threshold and codek in vs:
            false_counter += 1
            need_save += 1
        if need_save:
            f = open(f"false_case/false_{false_counter}.txt", "w")
            f.write(
                "does the requirement have relation with the code? Answer me in Chinese\n\n"
            )
            f.write(str(c))
            f.write("\n######################\n")
            f.write(codek)
            f.write("\n######################\n" + k)
precision = true_relevant / retrieved
recall = true_relevant / true_code
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {true_relevant}/{retrieved} = {precision*100:.2f}%")
print(f"Recall: {true_relevant}/{true_code} = {recall*100:.2f}%")
print(f"F1-Score: {f1_score*100:.2f}%")
