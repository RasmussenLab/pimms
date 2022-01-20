import pathlib
import logging
logger = logging.getLogger(__name__)

def _savefig(fig, name, folder:pathlib.Path ='.', pdf=True):
    """Save matplotlib Figure (having method `savefig`) as pdf and png."""
    folder = pathlib.Path(folder)
    fname = folder / name
    folder = fname.parent # in case name specifies folders
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(fname.with_suffix('.png'))
    if pdf: fig.savefig( fname.with_suffix('.pdf'))
    logger.info(f"Saved Figures to {fname}")


savefig = _savefig

