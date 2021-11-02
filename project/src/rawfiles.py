import os
from pathlib import Path, PurePosixPath

from IPython.display import display
import ipywidgets as widgets
import pandas as pd


queries = set()


def find_indices_containing_query(query, X):
    mask = X.index.str.contains(query)
    X_query = X.loc[mask].sort_index()
    queries.add(query)
    return X_query


def get_unique_stem(query, index: pd.Index):
    """Gets stem filename, by splitting filename left of query and remove last underscore _.
    
    Fractionated samples seem to be named by fraction type. Last field indicates fraction.
    """
    ret = index.str.split(query).str[0].str.rsplit("_", n=1).str[0]
    #     ret = index.str.rsplit('_', n=1).str[0]
    return sorted(list(set(ret)))


def show_fractions(stub: str, df):
    subset = df[df.index.str.contains(stub)]
    print(repr(stub))
    display(subset)
    display(f"N: {len(subset)}")


class RawFileViewer:
    def __init__(self, df:pd.DataFrame, start_query: str="[Ff]rac", outputfolder: str='.', path_col='path'):
        """Indices are used."""
        self.df = df
        self.file_names = df.index
        # self.queries = set() # add query button

        self.w_query = widgets.Text(start_query)
        self.query = start_query
        
        self.save_button = widgets.Button(description='Save current files.')
        self.save_button.on_click(self.save_current_files)

        self.w_data = widgets.Dropdown(
            options=self.get_options(self.w_query.value), index=0
        )
        self.stub = None
        self.folder = Path(outputfolder)
        self.path_col = path_col

    def get_options(self, query):
        # this needs to be clearer
        try:
            sub_df = self.find_indices_containing_query(query)
            ret = get_unique_stem(query, sub_df.index)
            return ret
        except:
            print(f"Not a valid query: {query} ")
            return ()
    
    def save_current_files(self, button):
        """Save files in current views as txt file.
        """
        folder = Path(self.folder) / self.query
        folder.mkdir(exist_ok=True)
        fname = folder / f"{self.stub}.txt"
        files = self.subset[self.path_col]
        line_template = "-get {remote_path} {local_path}"
        with open(fname, 'w') as f:
            f.write(f'-lmkdir {self.stub}\n')
            for _path in files:
                _local_path = PurePosixPath(self.stub)/_path.name
                _remote_path = PurePosixPath(_path)
                line = line_template.format(remote_path=_remote_path, local_path=_local_path)
                f.write(f'{line}\n')
        print(f"Saved file paths to: {fname}")
    
    def viewer(self, query, stub: str):
        if query != self.query:
            self.query = query
            print(f"updated query to: {query}")
            self.w_data.options = self.get_options(query)
            if len(self.w_data.options):
                stub = self.w_data.options[0]
            else:
                print(f"Nothing to display for QUERY: {query}")
                stub = None
            # find_indices_containing_query = partial(find_indices_containing_query, X=data_unique)
        if stub and stub!=self.stub:
            try:
                subset = self.df[self.df.index.str.contains(stub)]
                print('current stub: ', repr(stub))
                display(subset)
                display(f"N: {len(subset)}")
                self.subset = subset
            except TypeError:
                print(f"Nothing to display for query: {query}")
            self.stub = stub

    def find_indices_containing_query(self, query):
        mask = self.df.index.str.contains(query)
        X_query = self.df.loc[mask].sort_index()
        return X_query

    def view(self):
        """Interactive viewer. Updates list of options based on query."""
        # widget for type of data: meta or not. might be combined
        self.out_sel = widgets.interactive_output(
            self.viewer, {"stub": self.w_data, "query": self.w_query}
        )
        return widgets.VBox([self.w_query, self.w_data, self.out_sel, self.save_button])  # repr of class
