import os
import logging
logger = logging.getLogger(__name__)

def _savefig(fig, name, folder='.', pdf=True):
    """Save matplotlib Figure (having method `savefig`) as pdf and png."""
    filename = os.path.join(folder, name)
    fig.savefig(filename + '.png')
    if pdf: fig.savefig(filename + '.pdf')
    logger.info(f"Saved Figures to {filename}")