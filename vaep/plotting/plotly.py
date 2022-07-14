from functools import partial
import plotly.express as px

# set some defaults
from vaep.plotting import labels_dict, category_orders

TEMPLATE = 'seaborn'

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
                        width=1600,
                        height=700,
                        template=TEMPLATE
                        )


scatter = partial(px.scatter,
                  **scatter_defaults,
                  labels_dict=labels_dict,
                  category_orders=category_orders)

                  
bar = partial(px.bar,
             x='model',
             y='metric_value',
             color='data level',
             barmode="group",
             text='text',
             category_orders=category_orders,
             height=600,
             template=TEMPLATE)