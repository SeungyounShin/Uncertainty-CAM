from src.core.evaluators import ClsEvaluator, APEvaluator

EVALUATORS = {
    'top1_cls': ClsEvaluator,
    'ap': APEvaluator
}

def build(eval_config, data_config, logger):
    data_name = data_config['name']
    evaluators = {}
    if data_name == 'cub':
        evaluators['classification'] = EVALUATORS['top1_cls'](name='top1_cls', logger=logger)
    elif data_name == 'OpenImages30k':
        evaluators['classification'] = EVALUATORS['top1_cls'](name='top1_cls', logger=logger)
    elif data_name == 'mnist':
        evaluators['classification'] = EVALUATORS['top1_cls'](name='top1_cls', logger=logger)
    elif data_name == 'OxfordPets':
        evaluators['classification'] = EVALUATORS['top1_cls'](name='top1_cls', logger=logger)
    elif data_name == 'ImageNet':
        evaluators['classification'] = EVALUATORS['top1_cls'](name='top1_cls', logger=logger)
    elif data_name == 'voc':
        evaluators['classification'] = EVALUATORS['ap'](name='ap', logger=logger)

    logger.infov('Evaluators are built.')
    return evaluators
