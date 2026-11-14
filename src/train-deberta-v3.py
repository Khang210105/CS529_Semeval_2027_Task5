"""Fine-tune DeBERTa-v3-base for one five-fold repeat of EN-NN."""
from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/en-nn-train.tsv"
FOLDS_PATH = ROOT / "data/splits/en-nn_folds.csv"
MODEL_NAME = "microsoft/deberta-v3-base"
SEEDS = [42, 43, 44]
N_FOLDS = 5
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
MARKERS = ["<m>", "</m>", "<h>", "</h>"]
LABEL_COLS = ["ModAvg", "HeadAvg"]
ID_COL = "ContextID"

# Span matching rule used by the span-analysis cell in notebooks/full_pipeline_2.ipynb.
IRREGULAR_PLURALS = {
    "man": ["men"], "woman": ["women"], "child": ["children"],
    "person": ["people"], "mouse": ["mice"], "goose": ["geese"],
    "tooth": ["teeth"], "foot": ["feet"],
}


def word_forms(word: str) -> list[str]:
    w = str(word).strip().lower()
    if not w:
        return []
    forms = {w, *IRREGULAR_PLURALS.get(w, [])}
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        forms.add(w[:-1] + "ies")
    elif w.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(w + "es")
    else:
        forms.add(w + "s")
    if w.endswith("fe"):
        forms.add(w[:-2] + "ves")
    elif w.endswith("f"):
        forms.add(w[:-1] + "ves")
    return sorted(forms, key=len, reverse=True)

def find_compound_span(context: str, mod: str, head: str):
    """Find contiguous Mod + whitespace + Head, allowing common plurals."""
    mod_forms, head_forms = word_forms(mod), word_forms(head)
    if not mod_forms or not head_forms:
        return None
    mod_alt = "(?:" + "|".join(re.escape(x) for x in mod_forms) + ")"
    head_alt = "(?:" + "|".join(re.escape(x) for x in head_forms) + ")"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])(?P<mod>{mod_alt})(?P<gap>\s+)"
        rf"(?P<head>{head_alt})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    match = pattern.search(str(context))
    if match is None:
        return None
    return {
        "mod_start": match.start("mod"), "mod_end": match.end("mod"),
        "head_start": match.start("head"), "head_end": match.end("head"),
    }


def mark_context(row) -> str:
    sentence = str(row["Context"])
    span = find_compound_span(sentence, row["Mod"], row["Head"])
    if span is None:
        raise ValueError(
            f"Không dò được Mod + Head liền nhau: ContextID={row[ID_COL]}, "
            f"Compound={row['Compound']!r}"
        )
    return (
        sentence[:span["mod_start"]]
        + "<m>" + sentence[span["mod_start"]:span["mod_end"]] + "</m>"
        + sentence[span["mod_end"]:span["head_start"]]
        + "<h>" + sentence[span["head_start"]:span["head_end"]] + "</h>"
        + sentence[span["head_end"]:]
    )


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        marked_sentences = [mark_context(row) for _, row in frame.iterrows()]
        compounds = frame["Compound"].astype(str).tolist()
        encoded = tokenizer(
            marked_sentences,
            compounds,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.inputs = {key: value for key, value in encoded.items()}
        self.labels = torch.tensor(frame[LABEL_COLS].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        result = {key: value[index] for key, value in self.inputs.items()}
        result["labels"] = self.labels[index]
        return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_and_align(repeat: int) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(
        DATA_PATH, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False,
        dtype={ID_COL: str}, encoding="utf-8-sig",
    )
    folds = pd.read_csv(FOLDS_PATH, dtype={ID_COL: str}, encoding="utf-8-sig")
    fold_col = f"fold_r{repeat}"
    required = {ID_COL, "Compound", "Mod", "Head", "Context", *LABEL_COLS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {sorted(missing)}")
    if fold_col not in folds.columns:
        raise ValueError(f"Không có cột {fold_col} trong {FOLDS_PATH}")
    for name, frame in [("data", data), ("folds", folds)]:
        if frame[ID_COL].duplicated().any():
            raise ValueError(f"{name} có ContextID trùng.")
    data_ids, fold_ids = set(data[ID_COL]), set(folds[ID_COL])
    if data_ids != fold_ids:
        raise ValueError(
            f"ContextID trong data/folds khác nhau: thiếu {len(data_ids-fold_ids)}, "
            f"thừa {len(fold_ids-data_ids)}."
        )
    aligned = data.merge(
        folds[[ID_COL, "Compound", fold_col]],
        on=ID_COL, how="left", validate="one_to_one", suffixes=("", "_fold"),
    )
    if aligned[fold_col].isna().any():
        raise ValueError("Có ContextID không được gán fold.")
    if not aligned["Compound"].equals(aligned["Compound_fold"]):
        raise ValueError("Compound khác nhau giữa data và manifest folds.")
    aligned[fold_col] = pd.to_numeric(aligned[fold_col], errors="raise").astype(int)
    if not aligned[fold_col].isin(range(N_FOLDS)).all():
        raise ValueError(f"Giá trị {fold_col} phải nằm trong 0..{N_FOLDS-1}.")
    aligned[LABEL_COLS] = aligned[LABEL_COLS].apply(pd.to_numeric, errors="raise")
    return aligned, fold_col

def train_fold(train_df, valid_df, repeat, fold, device):
    seed = SEEDS[repeat]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    train_ds = PairDataset(train_df, tokenizer)
    valid_ds = PairDataset(valid_df, tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
    )
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, problem_type="regression",
    )
    model.resize_token_embeddings(len(tokenizer))
    # Keep trainable weights/gradients in FP32; autocast applies FP16 to eligible
    # forward operations, while GradScaler unscales the FP32 parameter gradients.
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * WARMUP_RATIO),
        num_training_steps=steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                output = model(**batch)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach().float().item()
        print(
            f"repeat={repeat} fold={fold} epoch={epoch}/{EPOCHS} "
            f"train_mse={loss_sum / len(train_loader):.6f}", flush=True,
        )

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in valid_loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                predictions.append(model(**batch).logits.float().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    result = pd.DataFrame({
        ID_COL: valid_df[ID_COL].to_numpy(),
        "pred_mod": pred[:, 0],
        "pred_head": pred[:, 1],
        "fold": fold,
    })
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True, choices=range(3))
    parser.add_argument(
        "--resume", action="store_true",
        help="resume using completed fold predictions in the partial OOF file",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cần GPU CUDA; trên Kaggle hãy bật Accelerator = GPU.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}; repeat={args.repeat}; seed={SEEDS[args.repeat]}", flush=True)

    data, fold_col = read_and_align(args.repeat)
    # Fail before downloading/training if the Việc 0 span rule misses any row.
    spans_missing = [
        row[ID_COL] for _, row in data.iterrows()
        if find_compound_span(row["Context"], row["Mod"], row["Head"]) is None
    ]
    if spans_missing:
        raise ValueError(
            f"Không dò được span ở {len(spans_missing)} dòng; ContextID mẫu: "
            f"{spans_missing[:10]}"
        )
    print(f"Span check: {len(data)}/{len(data)} câu có Mod + Head span.", flush=True)

    output_dir = ROOT / "outputs/oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / f"deberta_v3_base_r{args.repeat}.csv"
    partial_path = output_dir / f"deberta_v3_base_r{args.repeat}.partial.csv"
    runtime_path = output_dir / f"deberta_v3_base_r{args.repeat}_runtime.csv"
    all_predictions, runtimes = [], []
    if args.resume and partial_path.exists():
        previous = pd.read_csv(partial_path, dtype={ID_COL: str})
        if not {ID_COL, "pred_mod", "pred_head", "fold"}.issubset(previous.columns):
            raise ValueError(f"Partial OOF không hợp lệ: {partial_path}")
        all_predictions = [part.copy() for _, part in previous.groupby("fold", sort=True)]
        completed = set(previous["fold"].astype(int).unique())
        if not completed.issubset(set(range(N_FOLDS))):
            raise ValueError("Partial OOF có fold ngoài khoảng 0..4.")
        if previous[ID_COL].duplicated().any():
            raise ValueError("Partial OOF có ContextID trùng.")
        failed_spread = []
        for fold, part in previous.groupby("fold", sort=True):
            std_mod = part["pred_mod"].std()
            std_head = part["pred_head"].std()
            if std_mod < 0.3 or std_head < 0.3:
                failed_spread.append(
                    f"fold {int(fold)}: mod={std_mod:.4f}, head={std_head:.4f}"
                )
        if failed_spread:
            raise RuntimeError(
                "Partial OOF có fold không đạt std >= 0.3; không thể resume/ghi OOF "
                "hoàn chỉnh cho đến khi xử lý: " + "; ".join(failed_spread)
            )
        if runtime_path.exists():
            runtimes = pd.read_csv(runtime_path).to_dict(orient="records")
        print(f"Resume: đã có fold {sorted(completed)}", flush=True)
    else:
        completed = set()

    for fold in range(N_FOLDS):
        if fold in completed:
            continue
        valid_df = data.loc[data[fold_col] == fold].copy().reset_index(drop=True)
        train_df = data.loc[data[fold_col] != fold].copy().reset_index(drop=True)
        if valid_df.empty:
            raise ValueError(f"Validation r{args.repeat}/fold{fold} rỗng.")
        overlap = set(train_df["Compound"]) & set(valid_df["Compound"])
        if overlap:
            raise ValueError(f"Compound leakage ở fold {fold}: {list(overlap)[:5]}")
        print(
            f"Start fold={fold}: train={len(train_df)}, valid={len(valid_df)}, "
            f"GPU={gpu_name}", flush=True,
        )
        start = time.perf_counter()
        pred = train_fold(train_df, valid_df, args.repeat, fold, device)
        elapsed = time.perf_counter() - start
        runtimes.append({"repeat": args.repeat, "seed": SEEDS[args.repeat],
                         "fold": fold, "seconds": elapsed, "gpu": gpu_name})
        pd.DataFrame(runtimes).to_csv(runtime_path, index=False)
        all_predictions.append(pred)
        pd.concat(all_predictions, ignore_index=True).to_csv(partial_path, index=False)
        std_mod, std_head = pred["pred_mod"].std(), pred["pred_head"].std()
        print(
            f"Done fold={fold}: {elapsed/60:.2f} min; GPU={gpu_name}; "
            f"prediction std mod={std_mod:.4f}, head={std_head:.4f}", flush=True,
        )
        if elapsed > 20 * 60:
            raise RuntimeError(
                f"Fold {fold} mất hơn 20 phút. Đã ghi runtime; dừng trước fold kế tiếp "
                "để báo nhóm và thống nhất có tiếp tục hay không."
            )
        if std_mod < 0.3 or std_head < 0.3:
            raise RuntimeError(
                f"Độ lệch chuẩn dự đoán dưới 0.3 ở fold {fold}: "
                f"mod={std_mod:.4f}, head={std_head:.4f}. OOF chưa được ghi."
            )

    oof = pd.concat(all_predictions, ignore_index=True)
    if oof[ID_COL].duplicated().any() or set(oof[ID_COL]) != set(data[ID_COL]):
        raise ValueError("OOF cuối không khớp một-một với ContextID dữ liệu gốc.")
    oof.to_csv(oof_path, index=False)
    if partial_path.exists():
        partial_path.unlink()
    print(f"OOF saved: {oof_path}", flush=True)
    print(f"Fold runtimes saved: {runtime_path}", flush=True)


if __name__ == "__main__":
    main()
