import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'AlexNet':
            from .networks import AlexNet
            network = AlexNet(*args, **kwargs)
        elif network_name == 'SVM':
            from .networks import SVM
            network = SVM(*args, **kwargs)
        elif network_name == 'RegNet':
            from .networks import RegNet
            network = RegNet(*args, **kwargs)          
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name
    
