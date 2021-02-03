import numpy as np

weights = [[0.1, 0.1, -0.3],
           [0.1, 0.2, 0.0],
           [0.0, 1.3, 0.1]]


def w_sum(a, b):
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output


def vect_mat_mul(vect, matrix):
    assert (len(vect) == len(matrix))
    output = [0, 0, 0]
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])
    return output


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

alpha = 0.01

input = [toes[0], wlrec[0], nfans[0]]

true = [hurt[0], win[0], sad[0]]

pred = neural_network(input, weights)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
    error[i] = (pred[i] - true[i]) ** 2
    delta[i] = pred[i] - true[i]


def zeros_matrix(a, b):
    out = np.zeros((a, b)).tolist()
    return out


def outer_prod(vec_a, vec_b):
    """вычисление каждого приращения weight_delta
и коррекция каждого веса"""
    out = zeros_matrix(len(vec_a), len(vec_b))
    # out = []
    for elem_a in range(len(vec_a)):
        for elem_b in range(len(vec_b)):
            out[elem_a][elem_b] = vec_a[elem_a] * vec_b[elem_b]
    return out


weight_deltas = outer_prod(input, delta)

print(weight_deltas)
