from src.core.models.mln import MixtureLogitNetwork
from src.core.models import resnet, cnn
from transformers import ViTFeatureExtractor, ViTModel

MODELS = {
    'mln': MixtureLogitNetwork
}

BACKBONES = {
    'cnn' : cnn.cnn_builder,
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
}

def build(model_config, logger):
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    model_params = model_config['mol_params'].copy()
    backbone_params = model_config.get('backbone_params', None)

    # Initialize the backbone
    if backbone_params is not None:
        backbone_name = backbone_params.pop('name')

        backbone_params['num_classes'] = num_classes
        backbone_params['pretrained'] = True
        if backbone_name in BACKBONES:
            backbone = BACKBONES[backbone_name](**backbone_params)
        else:
            if backbone_name=='google/vit-base-patch16-224-in21k':
                feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
                backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            else:
                logger.error(
                    'Specify a valid backbone type among {}.'.format(BACKBONES.keys())
                ); exit()
    else:
        backbone = None

    # Initialize the model
    model_params['backbone'] = backbone
    try:
        model_params['feature_extractor'] = feature_extractor
    except:
        pass
    model_params['y_dim'] = num_classes
    model_params['multilabel'] = True

    if model_name in MODELS:
        model = MODELS[model_name](**model_params)
    else:
        logger.error(
            'Specify a valid model type among {}.'.format(MODELS.keys())
        ); exit()

    return model
