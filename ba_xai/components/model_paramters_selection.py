import json
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import callback_context
from .model_parameters import MODEL_PARAMETERS
from app_instance import app, PATHS
from dash import html


def generate_model_parameters(model_name):
    model_parameters = MODEL_PARAMETERS.get(model_name, {})
    return [
        create_parameter_input(model_name, name, info)
        for name, info in model_parameters.items()
    ]


def create_parameter_input(model_name, param_name, param_info):
    checkbox_id = {"type": "checkbox-param", "index": f"{model_name}-{param_name}"}
    input_id = {"type": "input-param", "index": f"{model_name}-{param_name}"}

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Checkbox(
                            id=checkbox_id,
                            className="parameter-checkbox",
                            value=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Label(
                            param_name, html_for=f"input-{model_name}-{param_name}"
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Input(
                            type="number",
                            id=input_id,
                            step=param_info["step"],
                            value=param_info["default"],
                            debounce=True,
                            disabled=False,
                            size="sm",
                        ),
                        width=4,
                    ),
                ],
                className="mb-1",
            ),
        ]
    )


@app.callback(
    [Output({'type': 'input-param', 'index': ALL}, 'value')],
    [Input({'type': 'input-param', 'index': ALL}, 'value'),
     Input({'type': 'checkbox-param', 'index': ALL}, 'value')],
    [State({'type': 'input-param', 'index': ALL}, 'id')]
)
def update_config(input_values, checkbox_checkeds, input_ids):
    param_names = [id_dict['index'] for id_dict in input_ids]
    corrected_values = []
    config = {}

    for i, full_name in enumerate(param_names):
        model_name, param_name = full_name.split('-')
        param_info = MODEL_PARAMETERS[model_name][param_name]
        min_val, max_val = param_info['min'], param_info['max']

        # Korrigiere den Eingabewert, falls er außerhalb des Bereichs liegt
        corrected_value = input_values[i]
        if corrected_value is not None:
            corrected_value = max(min_val, min(corrected_value, max_val))

        corrected_values.append(corrected_value)

        # Aktualisiere die Konfiguration nur für aktivierte Parameter
        if checkbox_checkeds[i]:
            config[param_name] = corrected_value

    # Speichere die Konfiguration
    with open(PATHS["config_path"], "w") as file:
        json.dump(config, file, indent=4)

    return [corrected_values]

