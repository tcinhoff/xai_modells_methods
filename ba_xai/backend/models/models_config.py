from backend.models import LGBMModel, LinearRegressionModel, XGBoostModel, SklearnGAM

MODELS = {
    "LR": {
        "label": "Linear Regression",
        "class": LinearRegressionModel,
        "compatible_methods": ["COEFFICIENTS", "PDP", "SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": True,
    },
    "GAM": {
        "label": "GAM",
        "class": SklearnGAM,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_noramlization": False,
    },
    "LGBM": {
        "label": "LGBM",
        "class": LGBMModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_noramlization": False,
    },
    "XGBoost": {
        "label": "XGBoost",
        "class": XGBoostModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_noramlization": False,
    },
}
