import numpy as np

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

# Выходные данные (результаты)
walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

# for iteration in range(20):
#     prediction = input.dot(weights)
#     error = (goal_prediction - prediction) ** 2
#     delta = prediction - goal_prediction
#     weights = weights - (alpha * (input * delta))
#     print("Error:" + str(error) + " Prediction:" + str(prediction))

for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]
        prediction = input.dot(weights)
        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error
        delta = prediction - goal_prediction
        weights = weights - (alpha * (input * delta))
        print("Prediction:" + str(prediction))
    print("Error:" + str(error_for_all_lights) + "\n")

# Как работает стохастический градиентный спуск? Как было показано в предыдущем
# примере, он выполняет прогноз и корректировку веса для каждого
# обучающего примера в отдельности. Иначе говоря, он берет первую комбинацию
# огней светофора и пытается спрогнозировать поведение пешеходов для
# нее, вычисляет weight_delta и корректирует веса. Затем переходит ко второй
# комбинации и так далее. Он многократно перебирает все данные из набора,
# пока не найдет комбинацию весов, которая хорошо прогнозирует все обучающие
# примеры.
