from .coefficients import get_coefficents_component
from .pdp import get_pdp_component
from .shap import get_shap_component
from .lime import get_lime_component
from backend.models.models_config import (
    LinearRegressionModel,
    LGBMModel,
    XGBoostModel,
    SklearnGAM,
)

XAI_METHODS = {
    "Coefficients": {
        "label": "Modellkoeffizienten",
        "function": get_coefficents_component,
        "compatible_models": [LinearRegressionModel],
    },
    "PDP": {
        "label": "Partial Dependence Plots",
        "function": get_pdp_component,
        "compatible_models": [LinearRegressionModel],
    },
    "SHAP": {
        "label": "SHAP",
        "function": get_shap_component,
        "compatible_models": [LinearRegressionModel, LGBMModel, XGBoostModel, SklearnGAM],
    },
    "LIME": {
        "label": "LIME",
        "function": get_lime_component,
        "compatible_models": [LinearRegressionModel, LGBMModel, XGBoostModel, SklearnGAM],
    },
}
