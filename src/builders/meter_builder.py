from src.core.meters import MultiMeter

def build(num_batches, logger):
    meter = MultiMeter(
        'loss meter', logger, ['mace','alea','epis','loss','acc'], num_batches, fmt=':f')

    logger.infov('Loss meter is built.')
    return meter
