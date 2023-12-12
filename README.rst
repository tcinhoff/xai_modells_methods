Also was steht an?
Diese Woche:
Integration von neuen Modellen (LR, XGB, GAM)
Integration von neuen Methoden (SHAP, LIME)
Integration von Configs
Schönes anzeigen von Methoden


Nächste Woche:
Refactoring und schönes UI


Zum Schreiben:
Inhaltsverzeichnis entsprechend anpassen

Was funktioniert aktuell noch nicht?
- Die Configs sind noch nicht integriert
LIME und SHAP funktionieren für LR und GAM nicht
LIME hat im allgemeinen noch starke Probleme, es wird nichts angezeigt und alte Graphen bleiben bestehen

Wichtige Code segmente:
lgb.LGBMRegressor(
            verbose=-1, **self.parameters, importance_type="gain"
        ) #self.parameters betrifft nur die Hyperparameter

wichtig ist: 
x, y, sample_weights = self._get_training_data(hourly_sales)
self.model.fit(x, y, sample_weight=sample_weights)




README:
Was wird erwartet?
Als Eingabe für die Daten werden nur csv Dateien akzeptiert.
Die Trainingsdaten müssen eine spalte date, welches den index angibt haben. Diese wird nur für intere Zwecke verwendet
Die Trainingsdaten müssen eine spalte yhat, welche die vorhergesagten Werte enthält haben.

Die Testdaten müssen eine spalte date, welches den index angibt haben. Diese wird nur für intere Zwecke verwendet
Die Testdaten dürfen keine spalte yhat haben und müssen sonst mit den Trainingsdaten spalten übereinstimmen.