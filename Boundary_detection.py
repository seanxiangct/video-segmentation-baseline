from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras import Input, Model


def boundary_sensitive_TCN(k):

    # shot feature extraction

    inputs = Input(shape=())
    model = inputs

    model = Bidirectional(LSTM(k,
                               dropout=0.25,
                               recurrent_dropout=0.25,
                               return_sequences=True))(model)

    model = TimeDistributed(Dense(2, activation='softmax'))(model)
    model = Model(inputs=inputs, outputs=model)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
