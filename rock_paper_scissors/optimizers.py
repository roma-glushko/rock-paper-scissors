from morty.config import ComponentFactory
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD, Adagrad
from tensorflow_addons.optimizers import AdamW

optimizer_factory: ComponentFactory = ComponentFactory('optimizer_factory')

optimizer_factory.register('adam', Adam)
optimizer_factory.register('adamw', AdamW)
optimizer_factory.register('nadam', Nadam)
optimizer_factory.register('adagrad', Adagrad)
optimizer_factory.register('rmsprop', RMSprop)
optimizer_factory.register('sgd', SGD)

# todo: try to use SAM optimizer