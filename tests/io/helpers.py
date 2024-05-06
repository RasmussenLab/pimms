import numpy as np
import pandas as pd


def create_DataFrame():
    data = np.arange(100).reshape(-1, 5)
    data = pd.DataFrame(data,
                        index=(f'row_{i:02}' for i in range(data.shape[0])),
                        columns=(f'feat_{i:02}' for i in range(data.shape[1]))
                        )
    return data
