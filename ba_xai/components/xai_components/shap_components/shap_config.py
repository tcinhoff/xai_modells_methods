from .shap_pdp import get_shap_pdp_component
from .shap_waterfall import get_shap_waterfall_component
from .shap_bar import get_shap_bar_component
from .shap_model_bar import get_shap_model_bar_component
from .shap_beeswarm import get_shap_beeswarm_component
from .shap_violin import get_shap_violin_component

SHAP_METHODS = {
    "SHAP-Waterfall": {
        "label": "SHAP Waterfall",
        "function": get_shap_waterfall_component,
    },
    "ShapPDP": {
        "label": "Shap Partial Dependence Plots",
        "function": get_shap_pdp_component,
    },
    "SHAP-Bar": {
        "label": "SHAP Bar",
        "function": get_shap_bar_component,
    },
    "SHAP-Model-Bar": {
        "label": "SHAP Model Bar",
        "function": get_shap_model_bar_component,
    },
    "SHAP-Beeswarm": {
        "label": "SHAP Beeswarm",
        "function": get_shap_beeswarm_component,
    },
    "SHAP-Violin": {
        "label": "SHAP Violin",
        "function": get_shap_violin_component,
    },
}