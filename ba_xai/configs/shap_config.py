from shap.plots import waterfall, bar, beeswarm, violin

SHAP_METHODS = {
    "SHAP-Waterfall": {
        "label": "SHAP Waterfall",
        "local": True,
        "function": waterfall,
    },
    "Shap-PDP": {
        "label": "SHAP PDP",
        "local": None,
        "function": None,
    },
    "SHAP-Bar": {
        "label": "SHAP Bar",
        "local": True,
        "function": bar,
    },
    "SHAP-Global-Bar-Test": {
        "label": "SHAP Global Bar for Test Data",
        "local": False,
        "function": bar,
    },
    "SHAP-Global-Bar-Train": {
        "label": "SHAP Global Bar for Train Data",
        "local": False,
        "function": bar,
    },
    "SHAP-Beeswarm": {
        "label": "SHAP Beeswarm",
        "local": False,
        "function": beeswarm,
    },
    "SHAP-Violin": {
        "label": "SHAP Violin",
        "local": False,
        "function": violin,
    },
}
