def safe_spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    return float(spearmanr(a, b).statistic)

def evaluate_one_target(data, pred, gold_col, target_col="Compound"):
    gold = data[gold_col].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(gold, pred)))
    rho = safe_spearman(gold, pred)

    tmp = pd.DataFrame({target_col: data[target_col].values, "gold": gold, "pred": pred})
    grouped = tmp.groupby(target_col).agg(
        gold_std=("gold", lambda x: np.std(x, ddof=0)),
        pred_std=("pred", lambda x: np.std(x, ddof=0)),
        n=("gold", "size")
    )
    grouped = grouped[grouped["n"] >= 2].copy()
    # Chuẩn hóa sai số số học rất nhỏ thành 0; tránh Spearman giả khi prediction hằng số.
    grouped.loc[grouped["pred_std"] <= 1e-12, "pred_std"] = 0.0
    grouped.loc[grouped["gold_std"] <= 1e-12, "gold_std"] = 0.0
    context_var_rho = safe_spearman(grouped["gold_std"], grouped["pred_std"])
    zero_pred_std = int((grouped["pred_std"] <= 1e-12).sum())
    return {
        "Spearman": rho,
        "Context_variance_rho": context_var_rho,
        "RMSE": rmse,
        "n_compounds_for_variance": len(grouped),
        "zero_pred_std_compounds": zero_pred_std,
    }

def evaluate_predictions(data, pred_mod, pred_head, model_name="model"):
    rows = []
    for label, pred in zip(LABEL_COLS, [pred_mod, pred_head]):
        metrics = evaluate_one_target(data, pred, label)
        rows.append({"model": model_name, "target": label, **metrics})
    return pd.DataFrame(rows)