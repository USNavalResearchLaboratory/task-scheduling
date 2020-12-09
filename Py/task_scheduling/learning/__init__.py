from . import environments, SL_policy
__all__ = ['environments', 'SL_policy']

import tensorflow
if tensorflow.version.VERSION[0] == '1':
    from . import RL_policy
    __all__.append('RL_policy')
