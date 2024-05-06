import io
from tokenize import group
import pandas as pd
from vaep.pandas import select_max_by

#                         m/z Protein group IDs         Intensity   Score
# Sequence     Charge
# YYYIPQYK     2      569.284              3745   147,680,000.000  83.801
# YYVTIIDAPGHR 3      468.914              2873 8,630,000,000.000 131.830 # -> select
#              3      468.914              2873   103,490,000.000  94.767 # -> discard
#              2      702.867              2873 2,458,400,000.000  70.028
# YYVLNALK     2      492.282              3521   147,430,000.000  58.687

data = """
Sequence,Charge,m/z,Protein group IDs,Intensity,Score
YYYIPQYK,2,569.2844,3745,147680000.0,83.801
YYVTIIDAPGHR,3,468.91386,2873,104490000.0,95.864
YYVTIIDAPGHR,3,468.91386,2873,8630000000.0,131.83
YYVTIIDAPGHR,3,468.91386,2873,103490000.0,94.767
YYVTIIDAPGHR,2,702.867151,2873,2458400000.0,70.028
YYVLNALK,2,492.28166,3521,147430000.0,58.687
"""

expected = """
Sequence,Charge,m/z,Protein group IDs,Intensity,Score
YYYIPQYK,2,569.2844,3745,147680000.0,83.801
YYVTIIDAPGHR,3,468.91386,2873,8630000000.0,131.83
YYVTIIDAPGHR,2,702.867151,2873,2458400000.0,70.028
YYVLNALK,2,492.28166,3521,147430000.0,58.687
"""


def test_select_max_by():
    index_columns = ["Sequence", "Charge"]
    selection_column = 'Score'

    df = pd.read_csv(io.StringIO(data))

    actual = select_max_by(df,
                           grouping_columns=index_columns,
                           selection_column=selection_column).set_index(index_columns)

    desired = pd.read_csv(io.StringIO(expected), index_col=index_columns)
    assert desired.equals(actual)
