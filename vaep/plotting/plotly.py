from functools import partial
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
                                    'batch_size_collab',
                                    'n_params'],
                        )


scatter = partial(px.scatter,
                  **scatter_defaults,
                  **figure_size_defaults,
                  labels_dict=labels_dict,
                  category_orders=order_categories)


bar = partial(px.bar,
              x='model',
              y='metric_value',
              color='data level',
              barmode="group",
              text='text',
              category_orders=order_categories,
              height=600,
              template=TEMPLATE)


line = partial(px.line,
               **figure_size_defaults,
               )


def apply_default_layout(fig):
    fig.update_layout(
        font={'size': 18},
        xaxis={'title': {'standoff': 15}},
        yaxis={'title': {'standoff': 15}})
    return fig
