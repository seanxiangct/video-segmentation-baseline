from keras import backend as K
from keras import Input, Model
from keras.layers import ZeroPadding1D, Conv1D, Cropping1D, SpatialDropout1D, Activation, Lambda, MaxPooling1D, \
    TimeDistributed, Dense, UpSampling1D, multiply, LSTM, add, Bidirectional, BatchNormalization


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])


def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', online=False,
           optimizer="rmsprop", activation='norm_relu',
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
             optimizer="rmsprop", activation='norm_relu',
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
        model = Bidirectional(LSTM(n_nodes[-i - 1], dropout=0.3, return_sequences=True))(model)
        if online: model = Cropping1D((0, conv_len // 2))(model)

        # model = SpatialDropout1D(0.3)(model)

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


def encoder_identify_block(input_tensor, n_nodes, conv_len, activation='norm_relu'):
    x = Conv1D(n_nodes, conv_len, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)

    x = SpatialDropout1D(0.3)(x)
    # residual shortcut
    x = add([x, input_tensor])

    x = MaxPooling1D(2)(x)

    return x


def decoder_identify_block(input_tensor, n_nodes, activation='norm_relu'):
    x = LSTM(n_nodes, dropout=0.3, return_sequences=True)(input_tensor)

    # residual shortcut
    x = add([x, input_tensor])

    x = SpatialDropout1D(0.3)(x)

    if activation == 'norm_relu':
        x = Activation('relu')(x)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = WaveNet_activation(x)
    else:
        x = Activation(activation)(x)

    # x = UpSampling1D(2)(x)

    return x


def residual_TCN_LSTM(n_nodes, conv_len, n_classes, n_feat, max_len,
                      loss='categorical_crossentropy', online=False,
                      optimizer="rmsprop", activation='norm_relu',
                      return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs

    # encoder
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        # convolution over the temporal dimension
        model = encoder_identify_block(model, n_nodes[i], conv_len, activation='norm_relu')
        if online: model = Cropping1D((0, conv_len // 2))(model)

    # decoder
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if online: model = ZeroPadding1D((conv_len // 2, 0))(model)
        # convolution over the temporal dimension
        model = decoder_identify_block(model, n_nodes[-i - 1], activation='norm_relu')
        if online: model = Cropping1D((0, conv_len // 2))(model)

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