import numpy as np
from keras import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D

import innvestigate

# Creat Keras Sequential Model
model = Sequential()
model.add(Embedding(input_dim=219, output_dim=8))
model.add(Conv1D(filters=64, kernel_size=8, padding="valid", activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation=None))

# Analyze model
model.predict(np.random.randint(1, 219, (1, 100)))  # [[0.04913538 0.04234646]]

analyzer = innvestigate.create_analyzer(
    "lrp.epsilon", model, neuron_selection_mode="max_activation", **{"epsilon": 1}
)
a = analyzer.analyze(np.random.randint(1, 219, (1, 100)))
print(a[0], a[0].shape)
