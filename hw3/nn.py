from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
input_length = 6
output_length = 3


def get_actor() -> Model:
    i = Input(shape=(input_length,))
    o = Dense(24, activation='relu')(i)
    o = Dense(8, activation='relu')(o)
    o = Dense(output_length)(o)
    o = LeakyReLU(alpha=0.1)(o)
    return Model(inputs=i, outputs=o)


def get_critic() -> Model:
    i = Input(shape=(input_length,))
    o = Dense(256, activation='relu')(i)
    o = Dense(64, activation='relu')(o)
    o = Dense(1)(o)
    return Model(inputs=i, outputs=o)
