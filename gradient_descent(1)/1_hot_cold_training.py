# Метод обучения «холодно/горячо» очень прост. После получения прогноза
# вычисляются еще два прогноза, в одном случае с немного увеличенным весом,
# а в другом — с немного уменьшенным. Затем производится изменение веса
# в том направлении, которое дало наименьшую ошибку. Многократное повторение
# этой процедуры в итоге уменьшило ошибку до 0.

weight = 0.5
input = 0.5
goal_prediction = 0.8

step_amount = 0.001  # Шаг изменения веса в каждой итерации

#Повторить обучение много раз, чтобы получить наименьшую ошибку
for iteration in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) ** 2
    print("Error:" + str(error) + " Prediction:" + str(prediction))
    up_prediction = input * (weight + step_amount)  # Попробовать увеличить!
    up_error = (goal_prediction - up_prediction) ** 2
    down_prediction = input * (weight - step_amount)  # Попробовать уменьшить!
    down_error = (goal_prediction - up_prediction) ** 2
    if (down_error < up_error):
        weight = weight - step_amount #ЕcЛи уменьшение дало лучший результат, уменьшить!
    if (down_error >= up_error):
        weight = weight + step_amount #Если увеличение дало лучший результат, увеличить!






