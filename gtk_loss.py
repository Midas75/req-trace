import os
import csv
import torch
import logging
import numpy as np
from torch import nn, Tensor
from typing import Literal
import torch.nn.functional as F
from sentence_transformers import losses,SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
)
from sentence_transformers.util.tensor import _convert_to_tensor
from sentence_transformers.similarity_functions import SimilarityFunction
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class CosineSimilarityLossFP16(losses.CosineSimilarityLoss):
    def __init__(
        self,
        m: SentenceTransformer,
        loss_fct: nn.Module = nn.MSELoss(),
        cos_score_transformation: nn.Module = nn.Identity(),
    ):
        super().__init__(m, loss_fct, cos_score_transformation)

    def compute_loss_from_embeddings(self, embeddings, labels):
        output = self.cos_score_transformation(
            torch.cosine_similarity(embeddings[0], embeddings[1])
        )
        return self.loss_fct(output.float(), labels.float().view(-1))


def pairwise_gtk_cos_sim(
    a: Tensor, b: Tensor, group_size: int = 16, top_k: int = 4
) -> Tensor:
    a = _convert_to_tensor(np.array(a))
    b = _convert_to_tensor(np.array(b))
    assert a.shape == b.shape, "Shape mismatch between a and b"
    batch, hidden_dim = a.size()
    assert (
        hidden_dim % group_size == 0
    ), f"hidden_dim {hidden_dim} not divisible by group_size {group_size}"
    num_groups = hidden_dim // group_size

    # reshape into groups
    a_groups = a.view(batch, num_groups, group_size)
    b_groups = b.view(batch, num_groups, group_size)

    # compute group-wise cosine similarity
    sim = F.cosine_similarity(a_groups, b_groups, dim=2)  # [batch, num_groups]

    # take top_k per sample
    topk_sim, _ = torch.topk(sim, k=top_k, dim=1)  # [batch, top_k]

    # average top_k
    return topk_sim.mean(dim=1)  # [batch]


class GroupTopkCosineSimilarityLoss(losses.CosineSimilarityLoss):
    def __init__(
        self,
        model,
        group_size: int = 16,
        top_k: int = 4,
        loss_fct=nn.MSELoss(),
        cos_score_transformation=nn.Identity(),
        count_idx: bool = True,
    ):
        super().__init__(model, loss_fct or nn.MSELoss(), cos_score_transformation)
        self.group_size = group_size
        self.top_k = top_k
        self.count_idx = count_idx
        if count_idx:
            self.idx_counter = dict[int, int]()

    def update_counter(self, topk_idx: torch.Tensor):
        topk_idx_flat = topk_idx.flatten().tolist()  # [batch*top_k]
        for idx in topk_idx_flat:
            if idx in self.idx_counter:
                self.idx_counter[idx] += 1
            else:
                self.idx_counter[idx] = 1

    def compute_loss_from_embeddings(self, embeddings, labels):
        # embeddings: list of 2 tensors, each shape [batch, hidden_dim]
        emb1, emb2 = embeddings
        batch, hidden_dim = emb1.size()
        assert (
            hidden_dim % self.group_size == 0
        ), f"Hidden dim {hidden_dim} not divisible by group size {self.group_size}"
        num_groups = hidden_dim // self.group_size

        # reshape: [batch, num_groups, group_size]
        emb1_groups = emb1.view(batch, num_groups, self.group_size)
        emb2_groups = emb2.view(batch, num_groups, self.group_size)

        # compute cosine similarity group-wise
        # sim: [batch, num_groups]
        sim = F.cosine_similarity(emb1_groups, emb2_groups, dim=2)

        # get top_k per sample
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=1)  # [batch, top_k]
        if self.count_idx:
            self.update_counter(topk_idx)
        # average top_k
        output = self.cos_score_transformation(topk_sim.mean(dim=1))  # [batch]

        return self.loss_fct(output.float(), labels.float().view(-1))


class GroupTopkCosineSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: (
            list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None
        ) = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: (
            Literal["float32", "int8", "uint8", "binary", "ubinary"] | None
        ) = None,
        truncate_dim: int | None = None,
        group_size: int = 16,
        top_k: int = 4,
    ):
        super().__init__(
            sentences1,
            sentences2,
            scores,
            batch_size,
            main_similarity,
            similarity_fn_names,
            name,
            show_progress_bar,
            write_csv,
            precision,
            truncate_dim,
        )
        self.group_size = group_size
        self.top_k = top_k
        self.similarity_fn_names = ["gtk_cosine"]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:"
        )

        embeddings1 = self.embed_inputs(model, self.sentences1)
        embeddings2 = self.embed_inputs(model, self.sentences2)
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        labels = self.scores

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "gtk_cosine": lambda x, y: pairwise_gtk_cos_sim(
                x, y, self.group_size, self.top_k
            ),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                scores = (
                    similarity_functions[fn_name](embeddings1, embeddings2)
                    .detach()
                    .cpu()
                    .numpy()
                )
                eval_pearson, _ = pearsonr(labels, scores)
                eval_spearman, _ = spearmanr(labels, scores)
                metrics[f"pearson_{fn_name}"] = eval_pearson
                metrics[f"spearman_{fn_name}"] = eval_spearman
                logger.info(
                    f"{fn_name.capitalize()}-Similarity:\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}"
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(
                csv_path,
                newline="",
                mode="a" if output_file_exists else "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                    ]
                    + [
                        metrics[f"{metric}_{fn_name}"]
                        for fn_name in self.similarity_fn_names
                        for metric in ["pearson", "spearman"]
                    ]
                )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(
                metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names
            )
            metrics["spearman_max"] = max(
                metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names
            )

            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "spearman_max"
            else:
                self.primary_metric = f"spearman_{self.similarity_fn_names[0]}"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
