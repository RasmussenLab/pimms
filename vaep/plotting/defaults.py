import logging
import seaborn as sns

logger = logging.getLogger(__name__)

# ! default seaborn color map only has 10 colors
# https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.color_palette("husl", N) to get N distinct colors
color_model_mapping = {
    'KNN': sns.color_palette()[0],
    'KNN_IMPUTE': sns.color_palette()[1],
    'CF': sns.color_palette()[2],
    'DAE': sns.color_palette()[3],
    'VAE': sns.color_palette()[4],
    'RF': sns.color_palette()[5],
    'Median': sns.color_palette()[6],
    'None': sns.color_palette()[7],
    'BPCA': sns.color_palette()[8],
    'MICE-CART': sns.color_palette()[9],

}
# other_colors = sns.color_palette()[8:]
other_colors = sns.color_palette("husl", 20)
color_model_mapping['IMPSEQ'] = other_colors[0]
color_model_mapping['IMPSEQROB'] = other_colors[1]
color_model_mapping['MICE-NORM'] = other_colors[2]
color_model_mapping['SEQKNN'] = other_colors[3]
color_model_mapping['QRILC'] = other_colors[4]
color_model_mapping['GSIMP'] = other_colors[5]
color_model_mapping['MSIMPUTE'] = other_colors[6]
color_model_mapping['MSIMPUTE_MNAR'] = other_colors[7]
color_model_mapping['TRKNN'] = other_colors[8]
color_model_mapping['SVDMETHOD'] = other_colors[9]
other_colors = other_colors[10:]


def assign_colors(models):
    i = 0
    ret_colors = list()
    for model in models:
        if model in color_model_mapping:
            ret_colors.append(color_model_mapping[model])
        else:
            pos = i % len(other_colors)
            ret_colors.append(other_colors[pos])
            i += 1
    if i > len(other_colors):
        logger.info("Reused some colors!")
    return ret_colors


labels_dict = {"NA not interpolated valid_collab collab MSE": 'MSE',
               'batch_size': 'bs',
               'n_hidden_layers': "No. of hidden layers",
               'latent_dim': 'hidden layer dimension',
               'subset_w_N': 'subset',
               'n_params': 'no. of parameter',
               "metric_value": 'value',
               'metric_name': 'metric',
               }

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'CF', 'DAE', 'VAE']}

IDX_ORDER = (['proteinGroups', 'aggPeptides', 'evidence'],
             ['median', 'interpolated', 'CF', 'DAE', 'VAE'])


ORDER_MODELS = ['RSN', 'median', 'interpolated',
                'CF', 'DAE', 'VAE',
                ]

l_colors_to_use_hex = ['#937860',  # seaborn brown
                       '#4c72b0',  # seaborn blue
                       '#dd8452',  # seaborn organe
                       '#55a868',  # seaborn green
                       '#c44e52',  # seaborn red
                       '#8172b3',  # seaborn violete/lila
                       ]

d_colors_to_use_hex = {k: v for k, v in zip(ORDER_MODELS, l_colors_to_use_hex)}
