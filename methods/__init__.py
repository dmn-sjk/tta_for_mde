# Dictionary to store registered adaptation methods
METHODS = {}

def register_method(name):
    """
    Decorator to register adaptation methods in the METHODS dictionary.
    
    Args:
        name (str): The name to register the method under.
    
    Returns:
        callable: The decorator function
    """
    def decorator(cls):
        method_name = name
        METHODS[method_name] = cls
        return cls
    return decorator

from .adadepth import AdaDepth
from .ssl_naive import SSLNaive
from .ssl_frozen import SSLFrozen
from .sup_frozen import SupFrozen
from .ssl_naive_supervised import SSLNaiveSupervised
from .svdp import SVDP
from .consistency import Consistency
from .cotta import CoTTA
from .contrastive import Contrastive
from .mae import MAE
from .feat_align import FeatAlign