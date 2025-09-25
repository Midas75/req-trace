from tqdm import tqdm
from random import shuffle, choice
from sentence_transformers import InputExample
from conf import datasets_path, openai_url, openai_apikey, openai_model
from datasets import Dataset
from openai import AsyncClient
from pickle import load, dump
from collections import defaultdict


def load_smos(with_negatives=True, neg_ratio=1) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(f"{datasets_path}/SMOS/ground.txt", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground.keys(), "loading smos"):
        k_content = open(
            f"{datasets_path}/SMOS/use_cases/{k}", "r", encoding="utf-8"
        ).read()
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = open(
                f"{datasets_path}/SMOS/source_code/{vi}", "r", encoding=encoding
            ).read()
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )
    return result


def load_itrust(
    with_negatives: bool = True, neg_ratio: float = 1
) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(f"{datasets_path}/iTrust/ground.txt", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground.keys(), "loading itrust"):
        k_content = (
            open(f"{datasets_path}/iTrust/use_cases/{k}", "r", encoding=encoding)
            .read()
            .encode("utf-8")
            .decode("utf-8")
        )
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = (
                open(f"{datasets_path}/iTrust/source_code/{vi}", "r", encoding=encoding)
                .read()
                .encode("utf-8")
                .decode("utf-8")
            )
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )

    return result


def load_libest_code(with_negatives=True, neg_ratio=1) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(
        f"{datasets_path}/LibEST/req_to_code_ground.txt", "r", encoding=encoding
    ) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground.keys(), "loading libest code"):
        k_content = open(
            f"{datasets_path}/LibEST/requirements/{k}", "r", encoding="utf-8"
        ).read()
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = open(
                f"{datasets_path}/LibEST/source_code/{vi}", "r", encoding=encoding
            ).read()
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )
    return result


def load_etour(with_negatives=True, neg_ratio=1) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(f"{datasets_path}/eTOUR/ground.txt", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground.keys(), "loading etour"):
        k_content = open(
            f"{datasets_path}/eTOUR/use_cases/{k}", "r", encoding=encoding
        ).read()
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = open(
                f"{datasets_path}/eTOUR/source_code/{vi}", "r", encoding=encoding
            ).read()
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )
    return result


def load_ebt_code(with_negatives=True, neg_ratio=1) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(f"{datasets_path}/EBT/code_ground.txt", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    reqs = dict[str, str]()
    for req_line in open(
        f"{datasets_path}/EBT/requirements.txt", "r", encoding=encoding
    ).readlines():
        req_id, req_content = req_line.split("\t")
        reqs[req_id] = req_content
    for k in tqdm(ground.keys(), "loading ebt"):
        k_content = reqs[k]
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = open(
                f"{datasets_path}/EBT/source_code/{vi}", "r", encoding=encoding
            ).read()
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )
    return result


def load_albergate(with_negatives=True, neg_ratio=1) -> list[InputExample]:
    ground = dict[str, list[str]]()
    result = list[InputExample]()
    encoding = "Windows 1252"

    # 读取 ground.txt
    with open(f"{datasets_path}/Albergate/ground.txt", "r", encoding=encoding) as f:
        for line in f.readlines():
            files = line.strip().split(" ")
            ground[files[0]] = files[1:]

    # 读取所有文本内容
    all_k_contents = dict()
    all_vi_contents = dict()
    for k in tqdm(ground.keys(), "loading albergate"):
        k_content = open(
            f"{datasets_path}/Albergate/requirements/{k}", "r", encoding=encoding
        ).read()
        all_k_contents[k] = k_content
        for vi in ground[k]:
            vi_content = open(
                f"{datasets_path}/Albergate/source_code/{vi}", "r", encoding=encoding
            ).read()
            all_vi_contents[vi] = vi_content

    # 构建正样本
    for k, v in ground.items():
        k_content = all_k_contents[k]
        for vi in v:
            vi_content = all_vi_contents[vi]
            result.append(InputExample(texts=[k_content, vi_content], label=1.0))

    # 构建负样本
    if with_negatives:
        vi_keys = list(all_vi_contents.keys())
        for k, v in ground.items():
            k_content = all_k_contents[k]
            for _ in range(
                int(neg_ratio * len(v))
            ):  # 每个正样本生成 neg_ratio 个负样本
                neg_vi = choice(vi_keys)
                while neg_vi in v:  # 避免选到正样本
                    neg_vi = choice(vi_keys)
                neg_vi_content = all_vi_contents[neg_vi]
                result.append(
                    InputExample(texts=[k_content, neg_vi_content], label=0.0)
                )
    return result


async def translate_into(
    system_prompt: str = "Translate the following content given by user into English and output the translated content directly without attaching any other content. If the given content is code, only translate comments",
    user_content: str = "这是一句中国话",
    openai_url: str = openai_url,
    openai_model: str = openai_model,
    openai_apikey: str = openai_apikey,
    extra_body: dict = {"thinking": {"type": "disable"}},
) -> str:
    client = AsyncClient(api_key=openai_apikey, base_url=openai_url)
    stream = await client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        extra_body=extra_body,
        stream=True,
    )
    result = []
    async for chunk in stream:
        result.append(chunk.choices[0].delta.content)
    return "".join(result)


def to_dataset(input_examples: list[InputExample]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "text1": example.texts[0],
                "text2": example.texts[1],
                "label": float(example.label),
            }
            for example in input_examples
        ]
    )


def to_pickle(
    train_examples: list[InputExample],
    valid_examples: tuple[list[str], list[str], list[float]],
):
    dump(train_examples, open("train.pkl", "wb"))
    dump(valid_examples, open("valid.pkl", "wb"))


def from_pickle() -> (
    tuple[list[InputExample], tuple[list[str], list[str], list[float]]]
):
    return load(open("train.pkl", "rb")), load(open("valid.pkl", "rb"))


def shuffle_split(
    input_examples: list[InputExample], ratio: float = 0.8, by_req: bool = True
) -> tuple[list[InputExample], tuple[list[str], list[str], list[float]]]:
    if not by_req:
        # 原来的随机打乱逻辑
        split_idx = int(ratio * len(input_examples))
        shuffled = input_examples.copy()
        shuffle(shuffled)
        train = shuffled[:split_idx]
        valid = shuffled[split_idx:]
    else:
        # 按 req 分组，保证同一个 req 不出现在两份
        req_to_examples = defaultdict(list)
        for ie in input_examples:
            req_to_examples[ie.texts[0]].append(ie)

        unique_reqs = list(req_to_examples.keys())
        shuffle(unique_reqs)
        split_idx = int(ratio * len(unique_reqs))
        train_reqs = set(unique_reqs[:split_idx])
        valid_reqs = set(unique_reqs[split_idx:])

        train = []
        valid = []
        for req in train_reqs:
            train.extend(req_to_examples[req])
        for req in valid_reqs:
            valid.extend(req_to_examples[req])

    # 构造验证集的输出格式
    valid_s1 = [ie.texts[0] for ie in valid]
    valid_s2 = [ie.texts[1] for ie in valid]
    valid_score = [ie.label for ie in valid]

    return train, (valid_s1, valid_s2, valid_score)


def translate_smos():
    pass


if __name__ == "__main__":
    neg_ratio = 4
    data = (
        load_smos(neg_ratio=neg_ratio)
        + load_itrust(neg_ratio=neg_ratio)
        + load_libest_code(neg_ratio=neg_ratio)
        + load_etour(neg_ratio=neg_ratio)
        + load_ebt_code(neg_ratio=neg_ratio)
        + load_albergate(neg_ratio=neg_ratio)
    )
    train_examples, valid_tuples = shuffle_split(data)
    to_pickle(train_examples, valid_tuples)
    from_pickle()
