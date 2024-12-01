import numpy
import scipy.special

# определение класса нейронной сети
class NeuralNetwork:

  # инициализировать нейронную сеть
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

    # задать количество узлов во входном, скрытом и выходном слое
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes

    # Матрицы весовых коэффициентов связей wih и who.
    # Весовые коэффициенты связей между узлом i и узлом j
    # следующего слоя обозначены как w__i__j:
    # wll w21
    # wl2 w22 и т.д.
    self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
    self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    # коэффициент обучения
    self.lr = learningrate

    # использование сигмоиды в качестве функции активации
    self.activation_function = lambda x: scipy.special.expit(x)

    pass

  # тренировка нейронной сети
  def train(self, inputs_list, targets__list):

    # преобразовать список входных значений в двухмерный массив
    inputs = numpy.array(inputs_list, ndmin=2).T
    targets = numpy.array(targets__list, ndmin=2).T

    # рассчитать входящие сигналы для скрытого слоя
    hidden_inputs = numpy.dot(self.wih, inputs)
    # рассчитать исходящие сигналы для скрытого слоя
    hidden_outputs = self.activation_function(hidden_inputs)

    # рассчитать входящие сигналы для выходного слоя
    final_inputs = numpy.dot(self.who, hidden_outputs)
    # рассчитать исходящие сигналы для выходного слоя
    final_outputs = self.activation_function(final_inputs)


    # Ошибка выходного слоя
    output_errors = targets - final_outputs
    # Ошибка скрытого слоя
    hidden_errors = numpy.dot(self.who.T, output_errors)

    # Обновление весов
    self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                            numpy.transpose(hidden_outputs))
    self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                            numpy.transpose(inputs))
    pass

  # опрос нейронной сети
  def query(self, inputs_list):
    # преобразовать список входных значений
    # в двухмерный массив
    inputs = numpy.array(inputs_list, ndmin=2).T
    # рассчитать входящие сигналы для скрытого слоя
    hidden_inputs = numpy.dot(self.wih, inputs)
    # # рассчитать исходящие сигналы для скрытого слоя
    hidden__outputs = self.activation_function(hidden_inputs)
    # # рассчитать входящие сигналы для выходного слоя
    final_inputs = numpy.dot(self.who, hidden__outputs)
    # # рассчитать исходящие сигналы для выходного слоя
    final_outputs = self.activation_function(final_inputs)

    return final_outputs