from morty.config import ComponentFactory
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.experimental import CosineDecay

scheduler_factory: ComponentFactory = ComponentFactory('scheduler_factory')


@scheduler_factory.register('exp_scheduler')
def get_exp_learning_rate_scheduler(initial_learning_rate: float, decay_rate: float = 0.95):
    return LearningRateScheduler(lambda epoch: initial_learning_rate * decay_rate ** epoch, verbose=True)


@scheduler_factory.register('cosine_scheduler')
def get_cosine_decay_scheduler(initial_learning_rate: float, decay_steps, alpha: float = 0.0):
    return LearningRateScheduler(CosineDecay(
        initial_learning_rate, decay_steps, alpha=alpha,
    ), verbose=True)


@scheduler_factory.register('piecewise_scheduler_with_warmups')
def get_piecewise_scheduler_with_warmups(
        batch_size: int,
        initial_learning_rate: float = 5e-6,
        min_learning_rate: float = 1e-6,
        max_learning_rate: float = 1e-5,
):
    # todo: add reference
    max_learning_rate = max_learning_rate * batch_size
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.4

    def schedule_learning_rate(epoch):
        if epoch < lr_ramp_ep:
            lr = (max_learning_rate - initial_learning_rate) / lr_ramp_ep * epoch + initial_learning_rate
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = max_learning_rate
        else:
            lr = (max_learning_rate - min_learning_rate) * lr_decay ** (
                    epoch - lr_ramp_ep - lr_sus_ep) + min_learning_rate
        return lr

    return LearningRateScheduler(schedule_learning_rate, verbose=True)
