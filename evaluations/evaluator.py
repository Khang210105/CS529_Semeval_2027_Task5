"""Local evaluator for SemEval-2027 Task 5 (ComPartMENT), Subtask A, English noun compounds.

Re-implements the three official metrics from the CodaBench task page. Each metric is
computed separately for the two settings of the subtask -- ``mod`` (gold column ModAvg)
and ``head`` (gold column HeadAvg) -- and then averaged over the two settings.

m1    Compositionality correlation (main): Spearman's rho between system and gold
      ratings over all sentences of a setting.
m2    Context variance correlation (secondary): for each target compound, the standard
      deviation of its ratings across its example sentences, once for the system and
      once for the gold; then Spearman's rho between the two lists (one value per compound).
rmse  Root mean squared error between system and gold ratings (complementary).

Run from the repository root:

    python -m src.evaluator --gold data/raw/en-nn-train.tsv --pred outputs/oof/b3_r0.csv
    python -m src.evaluator --gold data/raw/en-nn-train.tsv --pred outputs/oof/b3_r0.csv \\
        --folds data/splits/en-nn_folds_v1.csv --repeat 0
    python -m src.evaluator --check en-nn-pred.tsv --ids data/raw/en-nn-test.tsv --zip submission.zip

From Python:

    from src.evaluator import read_gold, read_predictions, score
    res = score(read_gold("data/raw/en-nn-train.tsv"), read_predictions("outputs/oof/b3_r0.csv"))
    res["avg"]["m1"]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ID = "ContextID"
GROUP = "Compound"
SETTINGS = {"mod": ("pred_mod", "ModAvg"), "head": ("pred_head", "HeadAvg")}
PRED_COLS = ["pred_mod", "pred_head"]
GOLD_COLS = ["ModAvg", "HeadAvg"]
METRICS = ("m1", "m2", "rmse")
SUBMISSION_NAME = "en-nn-pred.tsv"
SCALE = (0.0, 5.0)
CONST_TOL = 1e-8  # a compound whose predictions span less than this counts as constant


class EvaluatorWarning(UserWarning):
    """The result is computable but suspicious: a NaN metric, too few compounds, collapsed predictions."""


def _warn(message: str) -> None:
    warnings.warn(message, EvaluatorWarning, stacklevel=3)


# --------------------------------------------------------------------------- metrics


def spearman(x, y) -> float:
    """Spearman's rho with average ranks for ties; NaN when either side is constant.

    Uses scipy, i.e. the Pearson correlation of the ranks. The textbook shortcut
    1 - 6*sum(d^2) / (n(n^2 - 1)) is exact only without ties, and it returns a
    plausible-looking number for constant input, so it is never used here.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(x, y)[0])


def score_setting(df: pd.DataFrame, pred_col: str, gold_col: str, *, group_col: str = GROUP,
                  ddof: int = 1, name: str | None = None) -> dict:
    """Official metrics plus diagnostics for one setting (one prediction/gold column pair).

    ``df`` has one row per sentence. ``ddof`` is the delta degrees of freedom of the
    per-compound standard deviation used by m2. The task page does not say which one the
    official scorer uses, so check both before trusting a small m2 difference.
    """
    p = df[pred_col].to_numpy(dtype=float)
    y = df[gold_col].to_numpy(dtype=float)
    grouped = df.groupby(group_col, sort=False)

    # m1: one Spearman over every sentence of the setting.
    m1 = spearman(p, y)

    # m2: spread of each compound's ratings across its sentences, then Spearman across
    # compounds. A compound with a single sentence has no spread (NaN with ddof=1) and is dropped.
    spread = grouped[[pred_col, gold_col]].std(ddof=ddof).dropna()
    m2 = spearman(spread[pred_col], spread[gold_col])

    # rmse: error in rating points.
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))

    # Diagnostics, not official.
    sizes = grouped.size()
    pred_range = grouped[pred_col].max() - grouped[pred_col].min()
    means = grouped[[pred_col, gold_col]].mean()
    within = [spearman(g[pred_col], g[gold_col]) for _, g in grouped if len(g) >= 3]
    within = [r for r in within if not math.isnan(r)]
    result = {
        "m1": m1,
        "m2": m2,
        "rmse": rmse,
        "n_sentences": int(len(df)),
        "n_compounds": int(len(sizes)),
        "n_compounds_m2": int(len(spread)),
        "n_constant_compounds": int(((pred_range <= CONST_TOL) & (sizes >= 2)).sum()),
        "rho_between": spearman(means[pred_col], means[gold_col]),
        "rho_within_mean": float(np.mean(within)) if within else math.nan,
        "n_compounds_within": len(within),
    }
    _check_setting(result, name or gold_col)
    return result


def _check_setting(r: dict, label: str) -> None:
    if math.isnan(r["m1"]):
        _warn(f"[{label}] m1 is NaN: the predictions (or the gold) are constant.")
    n, k = r["n_compounds_m2"], r["n_constant_compounds"]
    if math.isnan(r["m2"]):
        if n < 2:
            why = "fewer than 2 compounds have two or more sentences"
        elif k >= n:
            why = "every compound gets one value for all its sentences (type-level prediction)"
        else:
            why = "the per-compound standard deviations are all equal on one side"
        _warn(f"[{label}] m2 is NaN: {why}.")
        return
    if n < 3:
        _warn(f"[{label}] m2 uses only {n} compounds; a rank correlation of so few points means nothing.")
    if k:
        _warn(f"[{label}] {k} of {r['n_compounds']} compounds get one value for all their sentences; "
              "this silently lowers m2.")


def score(gold: pd.DataFrame, pred: pd.DataFrame, *, ddof: int = 1) -> dict:
    """Score predictions against gold.

    ``gold`` needs ContextID, Compound, ModAvg, HeadAvg; ``pred`` needs ContextID, pred_mod,
    pred_head. Rows are matched on ContextID, never on row order. Returns
    ``{"mod": {...}, "head": {...}, "avg": {"m1": ..., "m2": ..., "rmse": ...}}``.
    A NaN metric stays NaN, and so does the average it enters.
    """
    df = align(gold, pred)
    result = {name: score_setting(df, pc, gc, ddof=ddof, name=name) for name, (pc, gc) in SETTINGS.items()}
    result["avg"] = {}
    for m in METRICS:
        values = [result[s][m] for s in SETTINGS]
        result["avg"][m] = float(np.mean(values))
        bad = [s for s, v in zip(SETTINGS, values) if math.isnan(v)]
        if bad:
            _warn(f"[avg] {m} is NaN because {' and '.join(bad)} {'is' if len(bad) == 1 else 'are'} NaN.")
    return result


def score_cv(gold: pd.DataFrame, oof: pd.DataFrame, folds: pd.DataFrame | None = None, *,
             repeat: int | None = None, ddof: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score out-of-fold predictions one fold at a time.

    The fold of each sentence comes from ``folds[f"fold_r{repeat}"]`` when a folds table is
    given (an OOF ``fold`` column, if present, must agree with it); otherwise from
    ``oof["fold"]``. Returns ``(per_fold, summary)`` where ``summary`` holds the mean and std
    over folds. The fold mean is the number to compare models with, because each fold, like
    the test set, is one model scored on compounds it never saw. Scoring all OOF rows in one
    call to ``score`` mixes the calibrations of five models; keep that for stacking.
    """
    _require(oof, [ID, *PRED_COLS], "oof")
    o = oof.copy()
    o[ID] = _ids(o[ID])
    _unique(o[ID], "oof")
    gold_ids = _ids(gold[ID])
    _same_ids(gold_ids, o[ID], "gold")

    if folds is not None:
        if repeat is None:
            raise ValueError("Pass repeat=k to choose the column fold_r<k> of the folds table.")
        col = f"fold_r{repeat}"
        _require(folds, [ID, col], "folds")
        f = pd.DataFrame({ID: _ids(folds[ID]).to_numpy(), "_fold": folds[col].to_numpy()})
        _unique(f[ID], "folds")
        o = o.merge(f, on=ID, how="left", validate="one_to_one")
        if o["_fold"].isna().any():
            raise ValueError(f"{int(o['_fold'].isna().sum())} OOF ContextIDs are not in the folds table.")
        if "fold" in o.columns:
            differ = o["fold"].astype(int).to_numpy() != o["_fold"].astype(int).to_numpy()
            if differ.any():
                raise ValueError(f"The OOF 'fold' column disagrees with {col} on {int(differ.sum())} rows; "
                                 "was this OOF made with another repeat?")
    else:
        _require(o, ["fold"], "oof (no folds table given)")
        if o["fold"].isna().any():
            raise ValueError("Some OOF rows have no fold.")
        o["_fold"] = o["fold"]
    o["_fold"] = o["_fold"].astype(int)

    rows = []
    for k, part in o.groupby("_fold", sort=True):
        in_fold = gold_ids.isin(set(part[ID])).to_numpy()
        r = score(gold[in_fold], part, ddof=ddof)
        row = {"fold": int(k), "n_compounds": r["mod"]["n_compounds"], "n_sentences": r["mod"]["n_sentences"]}
        for s in (*SETTINGS, "avg"):
            for m in METRICS:
                row[f"{s}_{m}"] = r[s][m]
        for s in SETTINGS:
            row[f"{s}_n_constant"] = r[s]["n_constant_compounds"]
        rows.append(row)
    per_fold = pd.DataFrame(rows).set_index("fold")
    summary = pd.DataFrame({"mean": per_fold.mean(skipna=False), "std": per_fold.std(skipna=False)}).T
    return per_fold, summary


# --------------------------------------------------------------------------- alignment helpers


def align(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Join gold and predictions on ContextID; anything but a one-to-one match is an error."""
    _require(gold, [ID, GROUP, *GOLD_COLS], "gold")
    _require(pred, [ID, *PRED_COLS], "pred")
    g = pd.DataFrame({ID: _ids(gold[ID]).to_numpy(), GROUP: gold[GROUP].to_numpy()})
    p = pd.DataFrame({ID: _ids(pred[ID]).to_numpy()})
    _unique(g[ID], "gold")
    _unique(p[ID], "pred")
    _same_ids(g[ID], p[ID], "gold")
    for c in GOLD_COLS:
        g[c] = _numbers(gold[c], f"gold {c}").to_numpy()
    for c in PRED_COLS:
        p[c] = _numbers(pred[c], f"pred {c}").to_numpy()
    return g.merge(p, on=ID, how="inner", validate="one_to_one")


def _ids(values: pd.Series) -> pd.Series:
    """ContextIDs are compared as text, so 10001 matches "10001" but "10001.0" does not."""
    return values.astype(str).str.strip()


def _examples(items: list) -> str:
    return f"(e.g. {', '.join(repr(i) for i in items[:5])})" if items else ""


def _require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; it has {list(df.columns)}.")


def _unique(ids: pd.Series, what: str) -> None:
    dup = ids[ids.duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{what} has {len(dup)} duplicate ContextIDs {_examples(dup)}.")


def _same_ids(expected: pd.Series, got: pd.Series, where: str) -> None:
    exp, have = set(expected), set(got)
    missing, extra = sorted(exp - have), sorted(have - exp)
    if missing or extra:
        raise ValueError(f"ContextIDs differ: {len(missing)} IDs of {where} have no prediction "
                         f"{_examples(missing)}; {len(extra)} predicted IDs are not in {where} {_examples(extra)}.")


def _numbers(values: pd.Series, what: str) -> pd.Series:
    out = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    bad = ~np.isfinite(out.to_numpy())
    if bad.any():
        examples = pd.Series(values)[bad].astype(str).tolist()
        raise ValueError(f"{what}: {int(bad.sum())} value(s) are missing or not finite numbers {_examples(examples)}.")
    return out


# --------------------------------------------------------------------------- files


def read_table(path) -> pd.DataFrame:
    """Read a task TSV (trial, train or test) the way the team agreed.

    ``quoting=QUOTE_NONE`` stops a stray double quote in a sentence from swallowing the
    following rows; ``keep_default_na=False`` stops words such as "null" or "nan"
    (null hypothesis, nan bread) from turning into missing values.
    """
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False,
                       dtype={ID: str}, encoding="utf-8-sig")


def read_gold(path) -> pd.DataFrame:
    """Read a task TSV that carries gold ratings (trial or train)."""
    df = read_table(path)
    _require(df, [ID, GROUP, *GOLD_COLS], str(path))
    return df


def read_predictions(path) -> pd.DataFrame:
    """Read predictions in either format.

    * submission format: no header, ``ContextID<TAB>mod<TAB>head`` per line;
    * a table whose header has ContextID, pred_mod, pred_head (``.csv`` = comma, otherwise
      tab), e.g. an OOF file; extra columns such as ``fold`` are kept.
    """
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    if first.split(sep)[0].strip().lower() == ID.lower():
        df = pd.read_csv(path, sep=sep, dtype={ID: str}, encoding="utf-8-sig")
        _require(df, [ID, *PRED_COLS], str(path))
        return df
    return read_submission(path)


def read_submission(path) -> pd.DataFrame:
    """Read a CodaBench submission file: no header, ``ContextID<TAB>mod<TAB>head`` per line."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if i == 1 and line.startswith("﻿"):
                raise ValueError(f"{path} starts with a UTF-8 BOM; CodaBench may read it as part of the "
                                 "first ContextID. Save the file as plain UTF-8.")
            line = line.rstrip("\r\n")
            if not line.strip():
                _warn(f"{path}:{i}: blank line skipped.")
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{i}: expected 3 tab-separated fields, found {len(parts)}: {line[:80]!r}")
            if i == 1 and parts[0].strip().lower() == ID.lower():
                raise ValueError(f"{path}: a submission file must not have a header line.")
            try:
                rows.append((parts[0].strip(), float(parts[1]), float(parts[2])))
            except ValueError:
                raise ValueError(f"{path}:{i}: predictions must be numbers, found "
                                 f"{parts[1]!r} and {parts[2]!r}.") from None
    if not rows:
        raise ValueError(f"{path} has no predictions.")
    return pd.DataFrame(rows, columns=[ID, *PRED_COLS])


def validate_submission(pred: pd.DataFrame, expected_ids) -> dict:
    """Check that ``pred`` has exactly one finite prediction pair per expected ContextID.

    Raises ValueError for anything CodaBench would reject or score wrongly. Values outside
    the 0-5 scale only warn and are kept as they are: the official scorer does not clip,
    and RMSE counts the error. Returns a short report.
    """
    _require(pred, [ID, *PRED_COLS], "pred")
    ids = _ids(pred[ID])
    _unique(ids, "pred")
    _same_ids(_ids(pd.Series(list(expected_ids))), ids, "the ID file")
    report = {"n_rows": int(len(pred))}
    lo, hi = SCALE
    for c in PRED_COLS:
        v = _numbers(pred[c], f"pred {c}")
        outside = int(((v < lo) | (v > hi)).sum())
        report[f"{c}_min"], report[f"{c}_max"], report[f"{c}_outside_scale"] = float(v.min()), float(v.max()), outside
        if outside:
            _warn(f"{c}: {outside} prediction(s) fall outside [0, 5]; they are kept, not clipped.")
    return report


def write_submission(pred: pd.DataFrame, path=SUBMISSION_NAME, *, decimals: int = 6) -> Path:
    """Write predictions in the CodaBench format: no header, tab-separated, '\\n' line ends.

    Six decimals avoid the artificial ties that heavy rounding would add to the rank metrics.
    """
    _require(pred, [ID, *PRED_COLS], "pred")
    ids = _ids(pred[ID])
    _unique(ids, "pred")
    mod = _numbers(pred["pred_mod"], "pred pred_mod").to_numpy()
    head = _numbers(pred["pred_head"], "pred pred_head").to_numpy()
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for cid, m, h in zip(ids, mod, head):
            fh.write(f"{cid}\t{m:.{decimals}f}\t{h:.{decimals}f}\n")
    return path


def make_submission_zip(tsv_path, zip_path="submission.zip") -> Path:
    """Zip the prediction file at the root of the archive, as CodaBench expects."""
    tsv_path = Path(tsv_path)
    if tsv_path.name != SUBMISSION_NAME:
        raise ValueError(f"CodaBench expects the file to be named {SUBMISSION_NAME}, not {tsv_path.name}.")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_path, arcname=SUBMISSION_NAME)
    return Path(zip_path)


# --------------------------------------------------------------------------- command line


def _f(v) -> str:
    return "nan" if v is None or math.isnan(v) else f"{v:.4f}"


def format_result(res: dict) -> str:
    lines = [f"{'setting':<8}{'m1':>8}{'m2':>8}{'rmse':>8}{'compounds':>11}{'constant':>10}"
             f"{'between':>9}{'within':>8}"]
    for s in SETTINGS:
        r = res[s]
        lines.append(f"{s:<8}{_f(r['m1']):>8}{_f(r['m2']):>8}{_f(r['rmse']):>8}{r['n_compounds']:>11}"
                     f"{r['n_constant_compounds']:>10}{_f(r['rho_between']):>9}{_f(r['rho_within_mean']):>8}")
    a = res["avg"]
    lines.append(f"{'avg':<8}{_f(a['m1']):>8}{_f(a['m2']):>8}{_f(a['rmse']):>8}")
    lines.append("m1: Spearman over sentences (main) | m2: Spearman of per-compound std | rmse: error in points")
    lines.append("constant: compounds with one value for all sentences | between: Spearman of compound "
                 "means | within: mean Spearman inside compounds")
    return "\n".join(lines)


def format_cv(per_fold: pd.DataFrame, summary: pd.DataFrame) -> str:
    cols = [f"{s}_{m}" for s in (*SETTINGS, "avg") for m in METRICS]
    fmt = {c: _f for c in cols}
    table = per_fold[["n_compounds", *cols]].to_string(formatters=fmt)
    summ = summary[cols].to_string(formatters=fmt)
    a = summary["avg_m1"]
    return (f"{table}\n\n{summ}\n\nCompare models on avg_m1 = {_f(a['mean'])} ± {_f(a['std'])} "
            "(mean ± std over folds).")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.evaluator",
                                 description="Local evaluator for SemEval-2027 Task 5, Subtask A, en-nn.")
    ap.add_argument("--gold", help="task TSV with gold columns (trial or train)")
    ap.add_argument("--pred", help="predictions: submission TSV (no header) or CSV/TSV with "
                                   "ContextID, pred_mod, pred_head")
    ap.add_argument("--folds", help="folds CSV from make_folds.py; scores fold by fold")
    ap.add_argument("--repeat", type=int, help="which fold_r<k> column of --folds to use")
    ap.add_argument("--ddof", type=int, default=1, choices=[0, 1],
                    help="ddof of the per-compound standard deviation in m2 (default 1)")
    ap.add_argument("--json", help="also write the result to this JSON file")
    ap.add_argument("--check", metavar="PRED_TSV", help="validate a submission file instead of scoring it")
    ap.add_argument("--ids", help="with --check: task TSV whose ContextIDs the submission must cover")
    ap.add_argument("--zip", help="with --check: also write the CodaBench zip to this path")
    args = ap.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("always", EvaluatorWarning)
        warnings.showwarning = lambda msg, *a, **k: print(f"WARNING: {msg}", file=sys.stderr)
        try:
            if args.check:
                if not args.ids:
                    ap.error("--check needs --ids")
                pred = read_submission(args.check)
                report = validate_submission(pred, read_table(args.ids)[ID])
                print(f"OK: {report['n_rows']} rows, one per ContextID of {args.ids}.")
                for c in PRED_COLS:
                    print(f"{c}: min {report[c + '_min']:.4f}, max {report[c + '_max']:.4f}, "
                          f"outside [0, 5]: {report[c + '_outside_scale']}")
                if args.zip:
                    print(f"Wrote {make_submission_zip(args.check, args.zip)} containing {SUBMISSION_NAME}.")
                return 0

            if not (args.gold and args.pred):
                ap.error("pass --gold and --pred, or --check")
            gold, pred = read_gold(args.gold), read_predictions(args.pred)
            if args.folds:
                if args.repeat is None:
                    ap.error("--folds needs --repeat")
                folds = pd.read_csv(args.folds, dtype={ID: str})
                per_fold, summary = score_cv(gold, pred, folds, repeat=args.repeat, ddof=args.ddof)
                print(format_cv(per_fold, summary))
                result = {"per_fold": per_fold.reset_index().to_dict(orient="records"),
                          "summary": summary.to_dict(orient="index")}
            else:
                result = score(gold, pred, ddof=args.ddof)
                print(format_result(result))
                if "fold" in pred.columns:
                    print("Note: one score over all rows. For CV, add --folds and --repeat to score fold by fold.")
            if args.json:
                Path(args.json).write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
            return 0
        except (ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
