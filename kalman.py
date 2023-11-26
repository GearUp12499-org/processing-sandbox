import numpy
import numpy as np
import matplotlib as plt

#
# [ p ]
# [ v ]
#
state = np.array([0, 2])
covar = np.array([[2, 2], [2, 2]])
STEP_SIZE = 0.1
PREDICTOR = np.array([
    [1, STEP_SIZE],
    [0, 1],
])
NOISE = np.array([
    [3, 3],
    [3, 3],
])
SENSOR_NOISE = np.array([
    [5, 2],
    [2, 5],
])
SENSOR_TRANSFORM = np.array([
    [0.2, 4],
    [4, 0.2]
])


def predict(controls: numpy.ndarray, control_mapping: numpy.ndarray):
    global state, covar, PREDICTOR, NOISE
    transposed = PREDICTOR.transpose()
    state = (PREDICTOR @ state) + (control_mapping @ controls)
    covar = (PREDICTOR @ covar @ transposed) + NOISE


def update(sensor_read: numpy.ndarray):
    global state, covar, SENSOR_NOISE, SENSOR_TRANSFORM
    k_prime = ((covar @ SENSOR_TRANSFORM.transpose())
               @ ((SENSOR_TRANSFORM @ covar @ SENSOR_TRANSFORM.transpose()) + SENSOR_NOISE))
    state = state + k_prime @ (sensor_read - (SENSOR_TRANSFORM @ state))
    covar = covar - k_prime @ SENSOR_NOISE @ covar


control_map = np.array([
    (STEP_SIZE ** 2) / 2,
    STEP_SIZE
])

states: dict[int, numpy.ndarray] = {}
step = 0

while step < 100:
    predict(np.array([1]), control_map)
    states[step] = state.copy()
    step += 1
    update(numpy.array([1]))
    states[step] = state.copy()
    step += 1


