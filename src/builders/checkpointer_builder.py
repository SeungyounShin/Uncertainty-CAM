from src.core.checkpointers import CustomCheckpointer

def build(save_dir, logger, model, optimizer, scheduler):
    checkpointer = CustomCheckpointer(
        save_dir, logger, model, optimizer, scheduler, standard='accuracy')
    return checkpointer

