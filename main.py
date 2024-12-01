from neural_network import NeuralNetwork

# Параметры сети
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

# Создание экземпляра сети
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Тренировка сети (пример)

n.train([1.0, 0.5, -1.5], [0.3, 0.7, 0.1])

# Опрос сети
result = n.query([1.0, 0.5, -1.5])
print("Результат опроса:", result)