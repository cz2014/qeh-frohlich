"""
qeh-frohlich: Frohlich electron-phonon coupling in van der Waals heterostructures.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def find_data_file(filename):
    """Locate a data file, checking current directory first, then package data.

    Parameters
    ----------
    filename : str
        Name of the file to find (e.g. 'BN-chi.npz').

    Returns
    -------
    str
        Absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the file is not found in either location.
    """
    local = Path(filename)
    if local.is_file():
        return str(local)

    packaged = DATA_DIR / filename
    if packaged.is_file():
        return str(packaged)

    raise FileNotFoundError(
        f"{filename} not found in current directory or package data ({DATA_DIR})"
    )


# Public API re-exports
from .qeh import make_heterostructure, Heterostructure
from .frohlich import BBset
from .scat_pp import scat, valley
