from ba_xai.backend.xai_methods.coefficients import Coefficients
from ba_xai.components.xai_components.pdp import get_pdp_component
from ba_xai.backend.xai_methods.lime import LIME
from ba_xai.components.xai_components.shap import get_shap_component
from ba_xai.configs.models_config import (
    LinearRegressionModel,
    LGBMModel,
    XGBoostModel,
    SklearnGAM,
)

XAI_METHODS = {
    "COEFFICIENTS": {
        "label": "Model Coefficients",
        "function": Coefficients.get_coefficents_component,
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
        "compatible_models": [
            LinearRegressionModel,
            LGBMModel,
            XGBoostModel,
            SklearnGAM,
        ],
    },
    "LIME": {
        "label": "LIME",
        "function": LIME.get_lime_Iframe,
        "compatible_models": [
            LinearRegressionModel,
            LGBMModel,
            XGBoostModel,
            SklearnGAM,
        ],
    },
}
