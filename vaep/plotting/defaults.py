labels_dict = {"NA not interpolated valid_collab collab MSE": 'MSE',
               'batch_size_collab': 'bs',
               'n_hidden_layers': "No. of hidden layers",
               'latent_dim': 'hidden layer dimension',
               'subset_w_N': 'subset',
               'n_params': 'no. of parameter',
               "metric_value": 'value',
               'metric_name': 'metric',
               }

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE']}

IDX_ORDER = (['proteinGroups', 'aggPeptides', 'evidence'],
             ['median', 'interpolated', 'collab', 'DAE', 'VAE'])
