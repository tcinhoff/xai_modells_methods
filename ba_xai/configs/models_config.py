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
        "can_handle_nan": False,
    },
    "GAM": {
        "label": "GAM",
        "class": SklearnGAM,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": False,
        "can_handle_nan": False,
    },
    "LGBM": {
        "label": "LGBM",
        "class": LGBMModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_noramlization": False,
        "can_handle_nan": True,
    },
    "XGBoost": {
        "label": "XGBoost",
        "class": XGBoostModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_noramlization": False,
        "can_handle_nan": True,
    },
    "GPR": {
        "label": "GPR",
        "class": GPRModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": True,
        "can_handle_nan": False,
    },
    "MLP": {
        "label": "MLP",
        "class": MLPModel,
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": False,
        "use_data_noramlization": False,
        "can_handle_nan": False,
    },
}

MODEL_PARAMETERS = {
    "LGBM": LGBMModel.LGBM_PARAMS,
    "XGBoost": XGBoostModel.XGBOOST_PARAMS,
}
