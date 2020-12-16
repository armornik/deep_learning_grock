import numpy as np

# Прогнозирование с несколькими выходами и одним входом
weights_num = np.array([0.1, 0.2, 0])


def neural_network_num(input, weights):
    """Прогнозирование с одним выходом Numpy"""
    pred_nums = input.dot(weights)
    return pred_nums


toes_num = np.array([8.5, 9.5, 9.9, 9.0]) #Текущее среднее число игр, сыгранных игроками
wlrec_num = np.array([0.65, 0.8, 0.8, 0.9]) #Текущая доля игр, окончившихся победой
nfans_num = np.array([1.2, 1.3, 0.5, 1.0]) # Число болельщиков (в миллионах)

print('Результаты прогнозирования с несколькими выходами и одним входом')
for i in range(0, 3):
    input_num = np.array([toes_num[i], wlrec_num[i], nfans_num[i]])
    pred_num = neural_network_num(input_num, weights_num)
    print(pred_num)


def ele_mul(number, vector):
    """Поэлементное умножение (реализация dot from Numpy)"""
    output = [0, 0, 0]
    assert (len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]

    return output


# Прогнозирование с несколькими входами и выходами
# Веса: травмы, победы, горечь от поражения
weights_sum = [[0.1, 0.1, -0.3],
                [0.1, 0.2, 0.0],
                [0.0, 1.3, 0.1]]


def w_sum(a, b):
    """Сумма произведений элементов векторов"""
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


def vect_mat_mul(vect, matrix):
    """Векторное умножение матрицы на число"""
    output = [0, 0, 0]
    assert (len(vect) == len(matrix))

    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])

    return output


def neural_network_sum(input, weights):
    """Вычисление для каждого выхода взвешенной суммы входов (вероятности события заложенного в весе)"""
    pred = vect_mat_mul(input, weights)
    return pred

toes = [8.5, 9.5, 9.9, 9.0] #Текущее среднее число игр, сыгранных игроками
wlrec = [0.65, 0.8, 0.8, 0.9] #Текущая доля игр, окончившихся победой
nfans = [1.2, 1.3, 0.5, 1.0] # Число болельщиков (в миллионах)

input = [toes[0], wlrec[0], nfans[0]]

print(weights_sum[0])
print(input)
pred = neural_network_sum((input, weights_sum[0]))
print('Получение прогноза с несколькими входами и выходами')
# print(pred)