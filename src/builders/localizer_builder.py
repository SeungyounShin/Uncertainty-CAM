from src.core.cams.fullgrad import FullGrad
from src.core.cams.gradcam import GradCAM
from src.core.cams.grad import InputGradient
from src.core.cams.simple_fullgrad import SimpleFullGrad
from src.core.cams.smooth_fullgrad import SmoothFullGrad
from src.core.cams.smoothgrad import SmoothGrad


LOCALIZERS = {
    'gradcam': GradCAM,
    #'full_grad': FullGrad,
    #'input_grad': InputGradient,
    #'simple_grad': SimpleFullGrad,
    #'smooth_grad': SmoothGrad,
    #'smooth_full_grad': SmoothFullGrad,
}

def build(model, criterion, eval_config, logger, device):
    localizer_name = eval_config.get('localizer', 'gradcam')
    loss_type = eval_config.get('loss_type', 'mace_avg')

    # Build a localizer
    if localizer_name in LOCALIZERS:
        localizer = LOCALIZERS[localizer_name](model, criterion, loss_type, device)
        logger.infov('Localizer is built.')
    else:
        logger.infov('Specify a valid localizer name among {}.'.format(LOCALIZERS.kyes()))

    return localizer
