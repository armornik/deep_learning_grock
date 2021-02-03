import numpy as np

np.random.seed(1)


# Возвращает x, если x > 0; иначе возвращает 0
def relu(x):
    return (x > 0) * x


# Возвращает 1, если output >0; иначе возвращает 0
def relu2deriv(output):
    return output >0


streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4
weights_0_l = 2 * np.random.random((3, hidden_size)) - 1
weights_l_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i + 1]
        layer_l = relu(np.dot(layer_0, weights_0_l))
        layer_2 = np.dot(layer_l, weights_l_2)
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i + 1]) ** 2)
        layer_2_delta = (walk_vs_stop[i:i + 1] - layer_2)
        layer_l_delta = layer_2_delta.dot(weights_l_2.T) * relu2deriv(layer_l)
        weights_l_2 += alpha * layer_l.T.dot(layer_2_delta)
        weights_0_l += alpha * layer_0.T.dot(layer_l_delta)

    # Эта строка вычисляет разность в слое 1ауег_1 с учетом разности в слое 1ауег_2, умножая layer_2_delta на
    # соответствующие веса weights_1 _2
    if iteration % 10 == 9:
        print("Error:" + str(layer_2_error))

# # Преобразовать отрицательные числа в 0
# def relu(x):
#     return (x > 0) * x
#
#
# alpha = 0.2
# hidden_size = 4
# walk_vs_stop = np.аrrау([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1]])
# streetlights = np.аrrау([[1, 1, 0, 0]]).T
#
# weights_0_l = 2 * np.random.random((3, hidden_size)) - 1
# weights_l_2 = 2 * np.random.random((hidden_size, 1)) - 1
#
# layer_0 = streetlights[0]
# layer_l = relu(np.dot(layer_0, weights_0_l))
# # Выход слоя 1ауег_1 пропускается через функцию relu,
# # которая превращает отрицательные значения в 0.
# # Он служит входом для следующего слоя, 1ауег2
# layer_2 = np.dot(layer_l, weights_l_2)
#
# # Порядок действий этого фрагмента
# # кода изображен на следующем рисунке. Входные данные поступают
# # в слой layerO. Посредством функции dot сигнал передается вверх
# # через веса из слоя 1ауег_0 в слой 1ауег_1 (вычисляются взвешенные
# # суммы для всех четырех узлов в слое 1ауег_1). Затем взвешенные
# # суммы из слоя 1ауег_1 передаются в функцию relu, которая преобразует
# # отрицательные числа в 0. И далее окончательные взвешенные суммы
# # попадают в последний слой 1ауег_2.
