weight = 0.1
alpha = 0.01


def neural_network(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5]

win_or_lose_binary = [1] # (победа!)

input = number_of_toes[0]

goal_pred = win_or_lose_binary[0]

pred = neural_network(input,weight)

# Делает чистую ошибку положительной,
# умножая ее на саму
# себя. Отрицательные ошибки
# не имеют смысла
error = (pred - goal_pred) ** 2

delta = pred - goal_pred  # Разность на выходе
# Здесь в delta записывается величина промаха. Истинный прогноз равен 1.0,
# а сеть вернула прогноз 0.85, то есть прогноз сети оказался на 0.15 меньше истины.
# Соответственно разность delta равна минус 0.15.

# Основное отличие этой реализации от градиентного спуска заключается в новой
# переменной delta. Она определяет чистую разность между прогнозом
# и истинным значением. Вместо непосредственного вычисления directionand_
# amount мы сначала находим величину, на которую отличается прогноз от
# истины, и только потом вычисляем direction_and_amount для изменения веса
# (в шаге 4, но теперь переменная weight переименована в weight_delta):
weight_delta = input * delta  # Разность весов

weight -= weight_delta * alpha  # Корректировка веса
