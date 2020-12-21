weights = [0.1, 0.2, -0.1]


def w_sum(a, b):
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output


def neural_network(inputs, weights):
    pred = w_sum(inputs, weights)
    return pred


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]

true = win_or_lose_binary[0]

input = [toes[0], wlrec[0], nfans[0]]

pred = neural_network(input, weights)

error = (pred - true) ** 2

delta = pred - true


def ele_mul(number,vector):
    """Вычисление всех приращений weight_delta и добавление их в каждый вес"""
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


weight_deltas = ele_mul(delta, input)

print(weight_deltas)

# 8.5 * -0.14 = -1.19 = weight-deltas[0]
# 0.65 * -0.14 = -0.091 = weight-deltas[1]
# 1.2 * -0.14 = -0.168 = weight-deltas[2]

alpha = 0.01

for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]

print(f'Weights: {weights}')
print(f'Weights deltas: {weight_deltas}')

# Weights: [0.1119, 0.20091, -0.09832]
# Weights deltas: [-1.189999999999999, -0.09099999999999994, -0.16799999999999987]


for iter in range(3):
    pred = neural_network(input,weights)
    error = (pred - true) ** 2
    delta = pred - true
    weight_deltas=ele_mul(delta,input)
    print("Iteration:" + str(iter+1))
    print("Pred:" + str(pred))
    print("Error:" + str(error))
    print("Delta:" + str(delta))
    print("Weights:" + str(weights))
    print("Weight_Deltas:")
    print(str(weight_deltas))
    print(
    )
    for i in range(len(weights)):
        weights[i]-= alpha * weight_deltas[i]
