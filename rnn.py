import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Exemplo de dados: Sequência simples de números
data = [[1, 2], [3, 4], [5, 6], [7, 8]]

# Transformando em formato apropriado para RNN
X = data[:-1]
y = data[1:]

# Criando o modelo RNN
model = Sequential()
model.add(SimpleRNN(10, input_shape=(len(X[0]), len(X[0][0]))))  # 10 unidades ocultas
model.add(Dense(2))  # Saída com 2 elementos

# Compilando o modelo
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
model.fit(X, y, epochs=50)

# Fazendo uma previsão
last_sequence = [7, 8]
prediction = model.predict([last_sequence])
print("Previsão com RNN:", prediction)
