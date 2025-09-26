from tqdm import tqdm
from random import shuffle, choice, sample
from sentence_transformers import InputExample
from conf import datasets_path, openai_url, openai_apikey, openai_model
from datasets import Dataset
from openai import AsyncClient
from pickle import load, dump
from collections import defaultdict


def load_general(
    path: str = "SMOS",
    ground: str = "ground.txt",
    key: str = "use_cases",
    value: str = "source_code",
    key_encoding: str = "Windows 1252",
    value_encoding: str = "Windows 1252",
) -> dict[str, dict[str, float]]:
    ground_dict = dict[str, list[str]]()
    result = dict[str, dict[str, float]]()
    encoding = "Windows 1252"
    with open(f"{datasets_path}/{path}/{ground}", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.encode().decode().strip().split(" ")
            ground_dict[files[0]] = files[1:]
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground_dict.keys(), f"Loading {path}"):
        k_content = (
            open(f"{datasets_path}/{path}/{key}/{k}", "r", encoding=key_encoding)
            .read()
            .encode()
            .decode()
        )
        all_k_contents[k] = k_content
        result[k_content] = dict[str, float]()
        for vi in ground_dict[k]:
            vi_content = (
                open(
                    f"{datasets_path}/{path}/{value}/{vi}", "r", encoding=value_encoding
                )
                .read()
                .encode()
                .decode()
            )
            all_vi_contents[vi] = vi_content
    for k, v in ground_dict.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result[k_content][vi_content] = 1.0
        for vi, vic in all_vi_contents.items():
            result[k_content].setdefault(vic, 0)
    return result


def load_smos() -> dict[str, dict[str, float]]:
    return load_general("SMOS", "ground.txt", "use_cases", "source_code", "utf-8")


def load_itrust() -> dict[str, dict[str, float]]:
    return load_general("iTrust", "ground.txt", "use_cases", "source_code")


def load_libest_code() -> dict[str, dict[str, float]]:
    return load_general(
        "LibEST", "req_to_code_ground.txt", "requirements", "source_code"
    )


def load_etour() -> dict[str, dict[str, float]]:
    return load_general("eTOUR", "ground.txt", "use_cases", "source_code")


def load_ebt_code() -> dict[str, dict[str, float]]:
    path = "EBT"
    ground = "code_ground.txt"
    ground_dict = dict[str, list[str]]()
    result = dict[str, dict[str, float]]()
    encoding = "Windows 1252"
    with open(f"{datasets_path}/{path}/{ground}", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground_dict[files[0]] = files[1:]
    all_k_contents = dict()
    all_vi_contents = dict()
    reqs = dict[str, str]()
    for req_line in open(
        f"{datasets_path}/{path}/requirements.txt", "r", encoding=encoding
    ).readlines():
        req_id, req_content = req_line.encode().decode().split("\t")
        reqs[req_id] = req_content
    for k in tqdm(ground_dict.keys(), f"Loading {path}"):
        k_content = reqs[k]
        all_k_contents[k] = k_content
        result[k_content] = dict[str, float]()
        for vi in ground_dict[k]:
            vi_content = (
                open(f"{datasets_path}/{path}/source_code/{vi}", "r", encoding=encoding)
                .read()
                .encode()
                .decode()
            )
            all_vi_contents[vi] = vi_content
    for k, v in ground_dict.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result[k_content][vi_content] = 1.0
        for vi, vic in all_vi_contents.items():
            result[k_content].setdefault(vic, 0)
    return result


def load_albergate() -> dict[str, dict[str, float]]:
    return load_general("Albergate", "ground.txt", "requirements", "source_code")


def load_all() -> list[dict[str, dict[str, float]]]:
    return [
        load_smos(),
        load_etour(),
        load_albergate(),
        load_ebt_code(),
        load_itrust(),
        load_libest_code(),
    ]


def split_by_req(
    datasets_l: list[dict[str, dict[str, float]]], train_ratio: float = 0.8
) -> tuple[list[dict[str, dict[str, float]]], list[dict[str, dict[str, float]]]]:
    all_req = set[str]()
    for dataset in datasets_l:
        for req in dataset:
            all_req.add(req)
    all_req = list(all_req)
    shuffle(all_req)
    split_idx = int(len(all_req) * train_ratio)
    train_req = set(all_req[:split_idx])
    # other_req = set(all_req[split_idx:])
    train_dataset = []
    other_dataset = []
    for dataset in datasets_l:
        t_ds = dict[str, dict[str, float]]()
        o_ds = dict[str, dict[str, float]]()
        for req in dataset:
            if req in train_req:
                t_ds[req] = dataset[req]
            else:
                o_ds[req] = dataset[req]
        train_dataset.append(t_ds)
        other_dataset.append(o_ds)
    return train_dataset, other_dataset


def to_pickle(
    train_dataset: list[dict[str, dict[str, float]]],
    other_dataset: list[dict[str, dict[str, float]]],
) -> None:
    dump(train_dataset, open("train.pkl", "wb"))
    dump(other_dataset, open("valid.pkl", "wb"))


def from_pickle() -> (
    tuple[list[dict[str, dict[str, float]]], list[dict[str, dict[str, float]]]]
):
    return load(open("train.pkl", "rb")), load(open("valid.pkl", "rb"))


def to_input_examples(
    datasets_l: list[dict[str, dict[str, float]]],
    negative_ratio: float = None,
) -> list[InputExample]:
    results = list[InputExample]()
    for dataset in datasets_l:
        for req, codes in dataset.items():
            avaliable_negcode = list[str]()
            for code, label in codes.items():
                if label > 0.5:
                    results.append(InputExample(texts=[req, code], label=label))
                else:
                    avaliable_negcode.append(code)
            if negative_ratio is not None and negative_ratio > 0:
                negative_count = int(len(codes) * negative_ratio)
                shuffle(avaliable_negcode)
                for nc in avaliable_negcode[:negative_count]:
                    results.append(InputExample(texts=[req, nc], label=0))
    return results

if __name__ == "__main__":
    all_dataset = load_all()
    td, od = split_by_req(all_dataset)
    to_pickle(td, od)
    td, od = from_pickle()
