from torch import optim

OPTIMIZERS = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def build(optim_config, model_params, logger):
    optimizer_name = optim_config.pop('name')
    optim_config['params'] = model_params

    if optimizer_name in OPTIMIZERS:
        optimizer = OPTIMIZERS[optimizer_name](**optim_config)
    else:
        logger.error(
            'Specify a valid optimizer name among {}.'.format(OPTIMIZERS.keys())
        ); exit()

    logger.infov('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer
