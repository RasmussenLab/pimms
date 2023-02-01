from functools import partial, update_wrapper
import plotly.express as px

# set some defaults
from .defaults import labels_dict, order_categories

TEMPLATE = 'none'


figure_size_defaults = dict(width=1600,
                            height=700,
                            template=TEMPLATE)

scatter_defaults = dict(x='data_split',
                        y='metric_value',
                        color="model",
                        facet_row="metric_name",
                        facet_col="subset_w_N",
                        hover_data=['n_hidden_layers',
                                    'hidden_layers',
                                    'batch_size',
                                    'n_params'],
                        )


scatter = partial(px.scatter,
                  **scatter_defaults,
                  **figure_size_defaults,
                  labels_dict=labels_dict,
                  category_orders=order_categories)
update_wrapper(scatter, px.scatter)


bar = partial(px.bar,
              x='model',
              y='metric_value',
              color='data level',
              barmode="group",
              text='text',
              category_orders=order_categories,
              height=600,
              template=TEMPLATE)
update_wrapper(bar, px.bar)


line = partial(px.line,
               **figure_size_defaults,
               )
update_wrapper(line, px.line)


def apply_default_layout(fig):
    fig.update_layout(
        font={'size': 18},
        xaxis={'title': {'standoff': 15}},
        yaxis={'title': {'standoff': 15}})
    return fig
