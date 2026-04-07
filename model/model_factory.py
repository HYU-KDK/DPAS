from model.clus_dpc import ClusDPC
from model.cluspro_baseline import ClusProBaseline
from model.cluspro_dual import ClusProDual
from model.adapt_dpc import AdaptDPC
from model.adapt_dpc_v2 import AdaptDPCv2

def get_model(config, attributes, classes, offset):
    if config.model_name == 'clus_dpc':
        model = ClusDPC(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'cluspro_baseline':
        model = ClusProBaseline(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'cluspro_dual':
        model = ClusProDual(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'adapt_dpc':
        model = AdaptDPC(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'adapt_dpc_v2':
        model = AdaptDPCv2(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(f"Unknown model: {config.model_name}")
    return model
