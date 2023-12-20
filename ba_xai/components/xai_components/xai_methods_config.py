from .coefficients import get_coefficents_component
from .pdp import get_pdp_component
from .shap import get_shap_component
from .lime import get_lime_component
from .shap_pdp import get_shap_pdp_component
from .shap_waterfall import get_shap_waterfall_component
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
    "ShapPDP": {
        "label": "Shap Partial Dependence Plots",
        "function": get_shap_pdp_component,
        "compatible_models": [LinearRegressionModel],
    },
    "SHAP": {
        "label": "SHAP",
        "function": get_shap_component,
        "compatible_models": [LinearRegressionModel, LGBMModel, XGBoostModel, SklearnGAM],
    },
    "SHAP-Waterfall": {
        "label": "SHAP Waterfall",
        "function": get_shap_waterfall_component,
        "compatible_models": [LinearRegressionModel, LGBMModel, XGBoostModel, SklearnGAM],
    },
    "LIME": {
        "label": "LIME",
        "function": get_lime_component,
        "compatible_models": [LinearRegressionModel, LGBMModel, XGBoostModel, SklearnGAM],
    },
}
