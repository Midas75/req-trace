from sentence_transformers import SentenceTransformer
from gtk_loss import pairwise_gtk_cos_sim
from sentence_transformers.util import pairwise_cos_sim
from typing import Callable
from tqdm import tqdm
from numpy import ndarray, array
from dataset import from_pickle
from conf_minilm import (
    sim_threshold,
    model_name,
    eval_topk,
    half,
    batch_size,
    use_gtk_cosine,
)


def do_eval(
    model: SentenceTransformer,
    batch_size: int,
    sim_threshold: float,
    top_k: int,
    sim_func: Callable[[ndarray, ndarray], float],
    valid_datasets: list[dict[str, dict[str, float]]],
) -> tuple[
    dict[str, int | float], dict[str, int | float], list[dict[str, dict[str, float]]]
]:
    model.eval()
    all_reqs = set[str]()
    all_codes = set[str]()
    for valid_dataset in valid_datasets:
        for req, codes in valid_dataset.items():
            all_reqs.add(req)
            for code, label in codes.items():
                # if label > sim_threshold:
                all_codes.add(code)
    all_strs = list(all_reqs) + list(all_codes)
    all_embs = dict[str, ndarray]()
    for i in tqdm(range(0, len(all_strs), batch_size), desc="encoding"):
        strs = all_strs[i : i + batch_size]
        embs = model.encode(strs, convert_to_numpy=True)
        for emb_idx, emb in enumerate(embs):
            all_embs[strs[emb_idx]] = emb
    model_predicts = []
    threshold_result: dict[str, int | float] = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "Accuracy": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "F2": 0.0,
    }
    topk_result = threshold_result.copy()
    for valid_dataset in tqdm(valid_datasets, desc="comparing"):
        model_predict = dict[str, dict[str, float]]()
        for req, codes in valid_dataset.items():
            model_predict[req] = dict[str, float]()
            req_code_scores = list[tuple[float, str]]()
            for code, label in codes.items():
                v = model_predict[req][code] = sim_func(
                    array([all_embs[req]]), array([all_embs[code]])
                )
                req_code_scores.append((float(v), code))
            req_code_scores = sorted(req_code_scores, key=lambda x: x[0], reverse=True)
            for i in range(0, len(req_code_scores)):
                score, code = req_code_scores[i]
                if code in codes and codes[code] > sim_threshold:
                    if i < top_k:
                        topk_result["TP"] += 1
                    else:
                        topk_result["FN"] += 1
                    if score > sim_threshold:
                        threshold_result["TP"] += 1
                    else:
                        threshold_result["FN"] += 1
                else:
                    if i < top_k:
                        topk_result["FP"] += 1
                    else:
                        topk_result["TN"] += 1
                    if score > sim_threshold:
                        threshold_result["FP"] += 1
                    else:
                        threshold_result["TN"] += 1
        model_predicts.append(model_predict)
    topk_result["TopK"] = top_k
    threshold_result["Threshold"] = sim_threshold
    for result in (topk_result, threshold_result):
        result["Accuracy"] = (result["TP"] + result["TN"]) / (
            result["TP"] + result["TN"] + result["FP"] + result["FN"]
        )
        result["Precision"] = result["TP"] / (result["TP"] + result["FP"])
        result["Recall"] = result["TP"] / (result["TP"] + result["FN"])
        result["F1"] = (
            2
            * result["Precision"]
            * result["Recall"]
            / (result["Precision"] + result["Recall"])
        )

        result["F2"] = (
            (1 + 2**2)
            * result["Precision"]
            * result["Recall"]
            / (2**2 * result["Precision"] + result["Recall"])
        )
    return threshold_result, topk_result, model_predicts


if __name__ == "__main__":
    m = SentenceTransformer(
        # model_name,
        "google/embeddinggemma-300m",
        # f"trained_qwen3/checkpoint-1214",
        # "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            # "attn_implementation": "flash_attention_2",
            "dtype": "half" if half else None,
            # "device_map": "auto",
        },
        tokenizer_kwargs={"padding_side": "left"},
    ).eval()
    td, od, ie = from_pickle()
    thr, tkr, mp = do_eval(
        m,
        batch_size,
        sim_threshold,
        eval_topk,
        pairwise_gtk_cos_sim if use_gtk_cosine else pairwise_cos_sim,
        od,
    )
    print("Threshold:  ", thr)
    print("Topk:       ", tkr)
