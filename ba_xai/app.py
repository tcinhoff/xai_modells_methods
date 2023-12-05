# ba_xai/app.py
from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Explainable AI Dashboard"),
    # Hier können weitere Dash-Komponenten für Ihre Anwendung hinzugefügt werden.
])

if __name__ == '__main__':
    app.run_server(debug=True)
