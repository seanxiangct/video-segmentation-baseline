from keras import backend as K, Sequential
from keras import Input, Model
from keras.layers import ZeroPadding1D, Conv1D, Cropping1D, SpatialDropout1D, Activation, Lambda, MaxPooling1D, \
    TimeDistributed, Dense, UpSampling1D, multiply, LSTM, add, Bidirectional, BatchNormalization, \
    GlobalAveragePooling1D, \
    Embedding, Conv2D, MaxPooling2D, Flatten, Dropout, Permute


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])



def attention_block(inputs):
    input_dim = int(inputs.shape[2])


def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', online=False,
           optimizer="rmsprop", activation='norm_relu',
           attention=True,
           return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))

    # attention layer, apply weightings to input
    if attention:
        # model = Permute((2, 1))(inputs)
        model = Dense(max_len, activation='softmax', name='attention_probs')(inputs)
        # model = Permute((2, 1), name='attention_vec')(model)
        model = multiply([inputs, model], name='attention_mul')
    else:
        model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        # convolution over the temporal dimension
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        # hidden features layer when in the last interation
        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[-i - 1], conv_len, padding='same')(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  sample_weight_mode="temporal",
                  metrics=['accuracy'])
    model.summary()

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if online:
            param_str += "_online"

        return model, param_str
    else:
        return model


def TCN_LSTM(n_nodes, conv_len, n_classes, n_feat, max_len,
             loss='categorical_crossentropy', online=False,
             optimizer="adam", activation='norm_relu',
             return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))

    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        # convolution over the temporal dimension
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        # hidden features layer when in the last interation
        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Bidirectional(LSTM(n_nodes[-i - 1],
                                   dropout=0.25,
                                   recurrent_dropout=0.25,
                                   return_sequences=True))(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  sample_weight_mode="temporal",
                  metrics=['accuracy'])
    model.summary()

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if online:
            param_str += "_online"

        return model, param_str
    else:
        return model


def attention_TCN_LSTM(n_nodes, conv_len, n_classes, n_feat, max_len,
                       loss='categorical_crossentropy', online=False,
                       optimizer="adam", activation='norm_relu',
                       attention=True,
                       return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))

    model = None
    # attention layer, apply weightings to input
    if attention:
        model = Permute((2, 1))(inputs)
        model = Dense(max_len, activation='softmax')(model)
        model = Permute((2, 1), name='attention_vec')(model)
        model = multiply([inputs, model], name='attention_mul')
    else:
        model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        # convolution over the temporal dimension
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        # hidden features layer when in the last interation
        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Bidirectional(LSTM(n_nodes[-i - 1],
                                   dropout=0.25,
                                   recurrent_dropout=0.25,
                                   return_sequences=True))(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  sample_weight_mode="temporal",
                  metrics=['accuracy'])
    model.summary()

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if online:
            param_str += "_online"

        return model, param_str
    else:
        return model


def lrcn(input_shape, n_classes):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py

    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556

    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
                                     activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, (3, 3),
                                     kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


def encoder_identify_block(input_tensor, n_nodes, conv_len):
    x = Conv1D(n_nodes, conv_len, padding='same')(input_tensor)
    x = BatchNormalization(axis=-1)(x)

    return x


def decoder_identify_block(input_tensor, n_nodes):
    x = LSTM(n_nodes, dropout=0.25, recurrent_dropout=0.25,
             return_sequences=True)(input_tensor)

    return x


def residual_TCN_LSTM(n_nodes, conv_len, n_classes, n_feat, max_len,
                      loss='categorical_crossentropy', online=False,
                      optimizer="rmsprop", depth=3,
                      return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs
    prev = Conv1D(n_nodes[0], conv_len, padding='same')(model)

    # encoder
    for i in range(n_layers):

        for j in range(depth):
            # convolution over the temporal dimension
            current = encoder_identify_block(prev, n_nodes[i], conv_len)
            # residual connection within residual block
            if j != 0:
                model = add([prev, current])
                model = Activation('relu')(model)
            prev = current

        if i < (n_layers - 1):
            model = MaxPooling1D(2)(model)

    # decoder
    # for i in range(n_layers):
    #
    #     for j in range(depth):
    #
    #         current = decoder_identify_block(model, n_nodes[-i - 1])
    #         model = add([prev, current])
    #         prev = current
    #     model = UpSampling1D(2)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(inputs=inputs, outputs=model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  sample_weight_mode="temporal",
                  metrics=['accuracy'])
    model.summary()

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if online:
            param_str += "_online"

        return model, param_str
    else:
        return model
