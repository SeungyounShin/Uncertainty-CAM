from src.core.criterions import CustomCriterion, MaceCriterion

CRITERIONS = {
    'custom': CustomCriterion,
    'mace': MaceCriterion
}

def build(train_config, model_config, data_config, device, logger):
    criterion_params = train_config['criterion']
    criterion_name = criterion_params.pop('name')

    data_name = data_config['name']
    if data_name == 'cub':
        is_multilabel = False
    elif data_name == 'OxfordPets':
        is_multilabel = False
    elif data_name == 'voc':
        is_multilabel = True
    elif data_name == 'mnist':
        is_multilabel = False
    elif data_name == 'OpenImages30k':
        is_multilabel = False
    elif data_name == 'ImageNet':
        is_multilabel = False
    criterion_params['is_multilabel'] = is_multilabel

    if criterion_name == 'mace':
        criterion_params['num_classes'] = model_config['num_classes']
        criterion_params['device'] = device

    criterion = CRITERIONS[criterion_name](**criterion_params)

    logger.infov('Criterion is built.')
    return criterion
