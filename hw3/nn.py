from tensorflow.keras.layers import Dense, Input, Reshape, Concatenate, Attention
from tensorflow.keras.models import Model
import os
input_length = 6
output_length = 3


# def get_actor() -> Model:
#     i = Input(shape=(input_length,))
#     o = Dense(256, activation='relu')(i)
#     o = Dense(64, activation='relu')(o)
#     o = Dense(output_length)(o)
#     o = LeakyReLU(alpha=0.1)(o)
#     return Model(inputs=i, outputs=o)
def get_actor() -> Model:
    i = Input(shape=(input_length,))
    o = Dense(36, activation='relu')(i)
    o = Dense(36, activation='relu')(o)
    o = Dense(output_length)(o)
    return Model(inputs=i, outputs=o)

def get_critic() -> Model:
    i = Input(shape=(input_length,))
    o = Dense(256, activation='relu')(i)
    o = Dense(64, activation='relu')(o)
    o = Dense(1)(o)
    return Model(inputs=i, outputs=o)


def get_progressive_actor(weight_path1, weight_path2):
    source_1 = get_actor()
    source_1.load_weights(os.path.join(weight_path1, 'actor.h5'))
    source_2 = get_actor()
    source_2.load_weights(os.path.join(weight_path2, 'actor.h5'))

    i = Input(shape=input_length)
    d1_1 = Dense(36, activation='relu', name='d1_1')
    d1_1.trainable = False
    o1_1 = d1_1(i)
    d1_2 = Dense(36, activation='relu', name='d1_2')
    d1_2.trainable = False
    o1_2 = d1_2(o1_1)

    d2_1 = Dense(36, activation='relu', name='d2_1')
    d2_1.trainable = False
    o2_1 = d2_1(i)

    v2_1 = Concatenate(axis=1)([Reshape(target_shape=(1, 36))(o2_1), Reshape(target_shape=(1, 36))(o1_1)])
    a2_1 = Attention()([Reshape(target_shape=(1, 36))(o2_1), v2_1])
    a2_1 = Reshape(target_shape=(36,))(a2_1)
    d2_2 = Dense(36, activation='relu', name='d2_2')
    d2_2.trainable = False
    o2_2 = d2_2(a2_1)

    d3_1 = Dense(36, activation='relu', name='d3_1')
    d3_1.trainable = False
    o3_1 = d3_1(i)

    v3_1 = Concatenate(axis=1)([Reshape(target_shape=(1, 36))(o3_1), Reshape(target_shape=(1, 36))(o2_1),
                               Reshape(target_shape=(1, 36))(o1_1)])
    a3_1 = Attention()([Reshape(target_shape=(1, 36))(o3_1), v3_1])
    a3_1 = Reshape(target_shape=(36,))(a3_1)
    d3_2 = Dense(36, activation='relu', name='d3_2')
    d3_2.trainable = False
    o3_2 = d3_2(a3_1)

    v3_2 = Concatenate(axis=1)([Reshape(target_shape=(1, 36))(o3_2), Reshape(target_shape=(1, 36))(o2_2),
                               Reshape(target_shape=(1, 36))(o1_2)])
    a3_2 = Attention()([Reshape(target_shape=(1, 36))(o3_2), v3_2])
    a3_2 = Reshape(target_shape=(36,))(a3_2)
    d3_3 = Dense(output_length, name='d3_3')
    d3_3.trainable = False
    o3_3 = d3_3(a3_2)
    model = Model(inputs=i, outputs=o3_3)

    for i in range(len(model.layers)):
        if model.layers[i].name == 'd1_1':
            model.layers[i].set_weights(source_1.layers[1].get_weights())
        elif model.layers[i].name == 'd1_2':
            model.layers[i].set_weights(source_1.layers[2].get_weights())
        elif model.layers[i].name == 'd2_1':
            model.layers[i].set_weights(source_2.layers[1].get_weights())
        elif model.layers[i].name == 'd2_2':
            model.layers[i].set_weights(source_2.layers[2].get_weights())

    return model


def get_progressive_critic(weight_path1, weight_path2):
    source_1 = get_critic()
    source_1.load_weights(os.path.join(weight_path1, 'critic.h5'))
    source_2 = get_critic()
    source_2.load_weights(os.path.join(weight_path2, 'critic.h5'))

    i = Input(shape=input_length)
    d1_1 = Dense(256, activation='relu', name='d1_1')
    d1_1.trainable = False
    o1_1 = d1_1(i)
    d1_2 = Dense(64, activation='relu', name='d1_2')
    d1_2.trainable = False
    o1_2 = d1_2(o1_1)

    d2_1 = Dense(256, activation='relu', name='d2_1')
    d2_1.trainable = False
    o2_1 = d2_1(i)

    v2_1 = Concatenate(axis=1)([Reshape(target_shape=(1, 256))(o2_1), Reshape(target_shape=(1, 256))(o1_1)])
    a2_1 = Attention()([Reshape(target_shape=(1, 256))(o2_1), v2_1])
    a2_1 = Reshape(target_shape=(256,))(a2_1)
    d2_2 = Dense(64, activation='relu', name='d2_2')
    d2_2.trainable = False
    o2_2 = d2_2(a2_1)

    d3_1 = Dense(256, activation='relu', name='d3_1')
    d3_1.trainable = False
    o3_1 = d3_1(i)

    v3_1 = Concatenate(axis=1)([Reshape(target_shape=(1, 256))(o3_1), Reshape(target_shape=(1, 256))(o2_1),
                               Reshape(target_shape=(1, 256))(o1_1)])
    a3_1 = Attention()([Reshape(target_shape=(1, 256))(o3_1), v3_1])
    a3_1 = Reshape(target_shape=(256,))(a3_1)
    d3_2 = Dense(64, activation='relu', name='d3_2')
    d3_2.trainable = False
    o3_2 = d3_2(a3_1)

    v3_2 = Concatenate(axis=1)([Reshape(target_shape=(1, 64))(o3_2), Reshape(target_shape=(1, 64))(o2_2),
                               Reshape(target_shape=(1, 64))(o1_2)])
    a3_2 = Attention()([Reshape(target_shape=(1, 64))(o3_2), v3_2])
    a3_2 = Reshape(target_shape=(64,))(a3_2)
    d3_3 = Dense(1, name='d3_3')
    d3_3.trainable = False
    o3_3 = d3_3(a3_2)
    model = Model(inputs=i, outputs=o3_3)

    for i in range(len(model.layers)):
        if model.layers[i].name == 'd1_1':
            model.layers[i].set_weights(source_1.layers[1].get_weights())
        elif model.layers[i].name == 'd1_2':
            model.layers[i].set_weights(source_1.layers[2].get_weights())
        elif model.layers[i].name == 'd2_1':
            model.layers[i].set_weights(source_2.layers[1].get_weights())
        elif model.layers[i].name == 'd2_2':
            model.layers[i].set_weights(source_2.layers[2].get_weights())

    return model