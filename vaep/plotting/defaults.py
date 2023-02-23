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

l_colors_to_use_hex = ['#937860', #seaborn brown
                       '#4c72b0', #seaborn blue
                       '#dd8452', #seaborn organe
                       '#55a868', #seaborn green
                       '#c44e52', #seaborn red
                       '#8172b3', #seaborn violete/lila
                       ]

d_colors_to_use_hex = {k: v for k, v in zip(ORDER_MODELS, l_colors_to_use_hex)}
