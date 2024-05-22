import numpy as np

from copy import deepcopy

from data_models import Detections


class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = np.eye(3)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def reset(self, initial_state):
        self.state = initial_state
        self.covariance = np.eye(3)

    def predict(self):
        # Assuming constant velocity model
        self.state = self.state
        self.covariance = self.covariance + self.process_noise

    def update(self, measurement):
        innovation = measurement - self.state
        innovation_covariance = self.covariance + self.measurement_noise
        kalman_gain = self.covariance @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (np.eye(3) - kalman_gain) @ self.covariance


kalman_filter = KalmanFilter(
    initial_state=np.zeros(3),
    process_noise=np.eye(3) * 0.01,
    measurement_noise=np.eye(3) * 0.1
)

x = Detections(camera_id=0, probability=0.9, timestamp=0, x=74.82153135636429, y=31.582607515414395, z=1.0)
temp = deepcopy(x)

# ********************

# TODO: Yep I should reinitialise the Kalman filter when I get a new detection after not having a detection for a while
#  like this

# ********************
# kalman_filter = KalmanFilter(
#     initial_state=np.array([x.x, x.y, x.z]),
#     process_noise=np.eye(3) * 0.01,
#     measurement_noise=np.eye(3) * 0.1
# )

for i in range(10):
    kalman_filter.predict()
    kalman_filter.update(np.array([x.x, x.y, x.z]))
    print(kalman_filter.state)
    print(kalman_filter.covariance)
    print()
    temp.x, temp.y, temp.z = kalman_filter.state
    print(temp)
#
# kalman_filter.predict()
# kalman_filter.update(np.array([x.x, x.y, x.z]))
# print(kalman_filter.state)
# print(kalman_filter.covariance)
# print()
#
# # How are these different even though this is just the first iteration?
#
# x.x, x.y, x.z = kalman_filter.state