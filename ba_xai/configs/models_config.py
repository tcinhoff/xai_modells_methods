from backend.models import (
    LGBMModel,
    LinearRegressionModel,
    XGBoostModel,
    SklearnGAM,
    GPRModel,
    MLPModel,    
)

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
        "config_upload": False,
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
    "GPR": {
        "label": "GPR",
        "class": GPRModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": True,
    },
    "MLP": {
        "label": "MLP",
        "class": MLPModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": False,
    },
}

MODEL_PARAMETERS = {
    "LGBM": LGBMModel.LGBM_PARAMS,
    "XGBoost": XGBoostModel.XGBOOST_PARAMS,
}
