def compute_metrics(y_true_mod, y_pred_mod, y_true_head, y_pred_head, compounds):
    # 1. Main metric: Overall Spearman Correlation
    spearman_mod, _ = spearmanr(y_true_mod, y_pred_mod)
    spearman_head, _ = spearmanr(y_true_head, y_pred_head)
    main_spearman = (spearman_mod + spearman_head) / 2.0
    
    # 2. Sub metric: Context Variance Correlation
    df_eval = pd.DataFrame({
        'comp': compounds,
        'true_mod': y_true_mod,
        'pred_mod': y_pred_mod,
        'true_head': y_true_head,
        'pred_head': y_pred_head
    })
    
    comp_var = df_eval.groupby('comp').agg({
        'true_mod': 'std', 'pred_mod': 'std',
        'true_head': 'std', 'pred_head': 'std'
    }).fillna(0)
    
    var_spearman_mod, _ = spearmanr(comp_var['true_mod'], comp_var['pred_mod'])
    var_spearman_head, _ = spearmanr(comp_var['true_head'], comp_var['pred_head'])
    sub_var_spearman = (np.nan_to_num(var_spearman_mod) + np.nan_to_num(var_spearman_head)) / 2.0
    
    # 3. Auxiliary metric: RMSE
    rmse_mod = np.sqrt(mean_squared_error(y_true_mod, y_pred_mod))
    rmse_head = np.sqrt(mean_squared_error(y_true_head, y_pred_head))
    avg_rmse = (rmse_mod + rmse_head) / 2.0
    
    return {
        'Spearman_Mod': spearman_mod,
        'Spearman_Head': spearman_head,
        'Main_Spearman': main_spearman,
        'Var_Spearman_Mod': var_spearman_mod,
        'Var_Spearman_Head': var_spearman_head,
        'Sub_Var_Spearman': sub_var_spearman,
        'RMSE_Mod': rmse_mod,
        'RMSE_Head': rmse_head,
        'Avg_RMSE': avg_rmse
    }
