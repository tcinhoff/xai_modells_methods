lgbm_params = {
    "num_leaves": {"default": 31, "min": 2, "max": 131070, "step": 1},
    "learning_rate": {"default": 0.01, "min": 0.00001, "max": 100000, "step": 0.00001},
    "n_estimators": {"default": 100, "min": 1, "max": 100000, "step": 1},
    "max_depth": {"default": -1, "min": -1, "max": 100000, "step": 1},
    "min_child_samples": {"default": 20, "min": 1, "max": 100000, "step": 1},
    "reg_alpha": {"default": 0.0, "min": 0.0, "max": 100000, "step": 0.00001},
    "reg_lambda": {"default": 0.0, "min": 0.0, "max": 100000, "step": 0.00001},
    "subsample": {"default": 1.0, "min": 0.00001, "max": 1.0, "step": 0.00001},
}

# gam_params = {
#     "spline_order": {"default": 3, "min": 1, "max": 5, "step": 1},
#     "n_splines": {"default": 25, "min": 10, "max": 50, "step": 1},
#     "lambda": {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1},
#     "max_iter": {"default": 100, "min": 50, "max": 500, "step": 50},
#     "tol": {"default": 1e-4, "min": 1e-5, "max": 1e-3, "step": 1e-5},
#     # 'scale': {"default": 1, "min": 0, "max": 1, "step": 0.1}  # Normalerweise automatisch berechnet
# }


xgboost_params = {
    "n_estimators": {"default": 100, "min": 1, "max": 100000, "step": 1},
    "learning_rate": {"default": 0.01, "min": 0.00001, "max": 100000, "step": 0.00001},
    "max_depth": {"default": 6, "min": 0, "max": 100000, "step": 1},
    "min_child_weight": {"default": 0, "min": 0, "max": 100000, "step": 1},
    "gamma": {"default": 0, "min": 0, "max": 100000, "step": 0.000001},
    "subsample": {"default": 1, "min": 0.0, "max": 1, "step": 0.000001},
    "colsample_bytree": {"default": 1, "min": 0.0, "max": 1, "step": 0.000001},
}


MODEL_PARAMETERS = {
    "LGBM": lgbm_params,
    "XGBoost": xgboost_params,
    # "GAM": gam_params,
}
