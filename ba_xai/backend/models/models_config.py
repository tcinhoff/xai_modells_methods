from backend.models import LGBMModel, LinearRegressionModel, XGBoostModel, SklearnGAM

MODELS = {
    "LR": {
        "label": "Linear Regression",
        "class": LinearRegressionModel,
        "compatible_methods": ["COEFFICIENTS", "PDP", "SHAP", "LIME"],
        "config_upload": False,
    },
    "GAM": {
        "label": "GAM",
        "class": SklearnGAM,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
    },
    "LGBM": {
        "label": "LGBM",
        "class": LGBMModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
    },
    "XGBoost": {
        "label": "XGBoost",
        "class": XGBoostModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
    },
}
