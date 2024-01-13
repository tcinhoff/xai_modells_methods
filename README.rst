Wie starte ich das ganze?  
Als erstes brauchen wir poetry install  
Dann sollte die umgebung mit source .venv/bin/activate aktiviert werde  
und dann kann man eigentlich schon mit poetry run python ba_xai/app.py starten   






Was fehlt noch?

GAM hat Probleme mit SHAP

Auslagern der Config mittels autoregistation und decoratorn:
# Decorator zur Registrierung der Klassen
def register_model(cls):
    if 'CONFIG' in cls.__dict__:
        ModelRegistry.register(cls.__name__, cls.CONFIG)
    return cls

class ModelRegistry:
    models = {}

    @classmethod
    def register(cls, name, config):
        cls.models[name] = config

# Beispiel für die Verwendung des Decorators
@register_model
class LGBMModel(BaseModel):
    CONFIG = {
        "label": "LGBM",
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_normalization": False
    }
    # Rest der Klasse...

# Zugriff auf die registrierten Modelle
print(ModelRegistry.models)


Scheint mir nicht wirkich intuitiv zu sein, auch wenn es eine coole Funktion zu scheinen ist.
Auch das auslagern der Config scheint mir nicht angebracht zu sein, da man sich dann nicht mehr so leicht am Muster orientieren kann und das zu fehlern führen würde
Auch müsste man weiterhin 2 Dateien anfassen

class LGBMModel(BaseModel):
    CONFIG = {
        "label": "LGBM",
        "compatible_methods": ["SHAP", "LIME"],
        "config_upload": True,
        "use_data_normalization": False
    }

    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)
        self.model = LGBMRegressor(verbose=-1, importance_type="gain") if config is None else LGBMRegressor(**config)

    # Rest der Methoden...

Neue Datei:

models = [LGBMModel, LinearRegressionModel, XGBoostModel, SklearnGAM]  # und weitere Modelle
config = {model.__name__: model.CONFIG for model in models}


Würde daher bei Configs bleiben