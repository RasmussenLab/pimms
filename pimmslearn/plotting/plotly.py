

def apply_default_layout(fig):
    fig.update_layout(
        font={'size': 18},
        xaxis={'title': {'standoff': 15}},
        yaxis={'title': {'standoff': 15}})
    return fig
