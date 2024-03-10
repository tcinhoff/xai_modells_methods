import plotly.graph_objs as go
from sklearn.inspection import partial_dependence


class PDP:
    def create_pdp_plot(model, feature, data):
        pdp_results = partial_dependence(model, data, [feature])
        feature_values = pdp_results["grid_values"][0]
        pdp_values = pdp_results["average"][0]

        trace = go.Scatter(x=feature_values, y=pdp_values, mode="lines", name="PDP")
        layout = go.Layout(
            title=f"Partial Dependence Plot for {feature}",
            xaxis={"title": feature},
            yaxis={"title": "Partial Dependence"},
        )
        return {"data": [trace], "layout": layout}
