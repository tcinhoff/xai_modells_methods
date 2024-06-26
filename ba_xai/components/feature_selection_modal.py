from dash import html, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import numpy as np
from app_instance import PATHS, app
from ba_xai.configs.models_config import MODELS
from dash.exceptions import PreventUpdate
import pandas as pd
import shap


def get_feature_selection_modal():
    feature_selection_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Select Features")),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    "SHAP values calculated with LGBM model for the test data"
                                ),
                                className="text-muted",
                                style={"fontSize": "0.9em", "marginTop": "10px"},
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save", id="close-modal", className="ml-auto"
                                ),
                                width={"size": "auto"},
                            ),
                        ],
                        justify="between",
                    ),
                ]
            ),
        ],
        id="feature-selection-modal",
        size="lg",
    )

    return html.Div(
        [
            feature_selection_modal,
            dcc.Store(id="selected-features-store"),
            dbc.Button(
                "Select Features",
                id="feature-reduction-button",
                n_clicks=0,
                style={"marginTop": "15px"},
            ),
        ]
    )


@app.callback(
    [
        Output("feature-selection-modal", "is_open"),
        Output("modal-body", "children"),
        Output("selected-features-store", "data"),
    ],
    [Input("feature-reduction-button", "n_clicks"), Input("close-modal", "n_clicks")],
    [
        State("selected-features-store", "data"),
        State({"type": "dynamic-checkbox", "index": ALL}, "value"),
        State({"type": "dynamic-checkbox", "index": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def toggle_modal(
    n_clicks, n_clicks_close, selected_features_data, checkbox_values, checkbox_ids
):
    ctx = callback_context
    if not ctx.triggered or (not n_clicks and not n_clicks_close):
        raise PreventUpdate

    train_data = pd.read_csv(PATHS["train_data_path"])
    test_data = pd.read_csv(PATHS["test_data_path"])

    if train_data.empty or test_data.empty:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "feature-reduction-button":
        feature_importance = train_lgbm_and_calculate_global_shap_values(
            train_data, test_data
        )

        content_children = [
            dbc.Row(
                [
                    dbc.Col(width=1),
                    dbc.Col(html.Div("Select"), width=2),
                    dbc.Col(html.Div("Feature Name"), width=6),
                    dbc.Col(html.Div("SHAP Value"), width=3),
                ],
                className="feature-selection-header",
            ),
        ]
        selected_features = (
            selected_features_data
            if selected_features_data
            else [row["feature"] for index, row in feature_importance.iterrows()]
        )

        for index, row in feature_importance.iterrows():
            is_selected = row["feature"] in selected_features
            content_children.append(
                create_feature_selection_row(
                    row["feature"], row["mean_shap_value"], is_selected
                )
            )

        content = html.Div(
            content_children, style={"maxHeight": "500px", "overflowY": "scroll"}
        )

        return [True, content, no_update]

    elif button_id == "close-modal":
        updated_selected_features = [
            checkbox_id["index"]
            for checkbox_value, checkbox_id in zip(checkbox_values, checkbox_ids)
            if checkbox_value
        ]
        selected_features_data = updated_selected_features

        return [
            False,
            no_update,
            selected_features_data,
        ]

    return [False, no_update, no_update]


def create_feature_selection_row(feature_name, shap_value, is_selected):
    return dbc.Row(
        [
            dbc.Col(width=1),
            dbc.Col(
                dbc.Checkbox(
                    id={"type": "dynamic-checkbox", "index": feature_name},
                    className="big-checkbox",
                    value=is_selected,
                ),
                width=2,
            ),
            dbc.Col(
                html.Label(feature_name, htmlFor=f"checkbox-{feature_name}"), width=6
            ),
            dbc.Col(html.Div(f"{shap_value:.2f}", className="shap-value"), width=3),
        ],
        className="feature-selection-row",
    )


def train_lgbm_and_calculate_global_shap_values(train_data, test_data):
    # Code zum Trainieren des LGBM-Modells und Berechnen der SHAP-Werte
    train_data = train_data.drop(columns=["date"])
    test_data = test_data.drop(columns=["date"])
    target_col = list(set(train_data.columns) - set(test_data.columns))[0]

    model_class = MODELS["LGBM"]["class"]
    model = model_class(train_data, target_col)
    model.fit()
    processed_train_data = train_data.drop(columns=[target_col])
    explainer = shap.Explainer(model.model, processed_train_data)
    shap_values = explainer(test_data)
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    feature_names = test_data.columns
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "mean_shap_value": mean_shap_values}
    )

    feature_importance = feature_importance.sort_values(
        by="mean_shap_value", ascending=False
    )

    return feature_importance
