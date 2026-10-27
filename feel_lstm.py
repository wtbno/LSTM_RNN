import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Carregando o dataset IMDB (sentiment analysis)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pré-processamento dos dados
maxlen = 500
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Criando o modelo LSTM
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Avaliando o modelo
score = model.evaluate(x_test, y_test)
print("Test accuracy:", score[1])
