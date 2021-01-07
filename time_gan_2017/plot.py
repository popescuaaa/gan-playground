import plotly.graph_objects as go


def plot_time_series(ts):
    time = [i for i in range(len(ts))]
    fig = go.Figure(data=go.Scatter(x=time, y=ts))
    fig.update_layout(
        autosize=False,
        width=230,
        height=230,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",
    )
    fig.show()
