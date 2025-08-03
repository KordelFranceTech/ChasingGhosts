import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):

    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.A = np.matrix([[1, self.dt],
                            [0, 1]])
        self.B = np.matrix([[(self.dt ** 2) / 2], [self.dt]])
        self.C = np.matrix([[1, 0]])
        # Process noise (modeling error) covariance matrix
        self.Q = np.matrix([[(self.dt ** 4) / 4, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, self.dt ** 2]]) * self.std_acc ** 2
        # Measurement noise covariance matrix
        self.R = std_meas ** 2
        # Estimation uncertainty (error) covariance matrix
        self.P = np.eye(self.A.shape[1])
        self.x = np.matrix([[0], [0]])


    def predict(self):
        # Ref :Eq.(9) and Eq.(10)
        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        # self.P = np.dot(self.A, self.P) + np.dot(self.P, self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        # S = C*P*C'+R //ALIGN WITH NOTES
        S = np.dot(self.C, np.dot(self.P, self.C.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        # L = P * C'* inv(C*P*C'+R) //ALIGN WITH NOTES
        """
        Kalman gain provides a weighting specifying how much it trusts the measurement versus the current state estimate.
        - If Kalman gain is large (C*P*C' >> R), then the filter trusts the current measurements over the current estimates.
        - If Kalman gain is small (C*P*C' << R), then rhe filter trusts the current estimates more than the current measurements.
        """
        L = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))  # Eq.(11)

        # Updated state estimation matrix
        self.x = np.round(self.x + np.dot(L, (z - np.dot(self.C, self.x))))  # Eq.(12)

        I = np.eye(self.C.shape[1])
        # Updated error covariance matrix
        self.P = (I - (L * self.C)) * self.P  # Eq.(13)


def build_kalman_filter(t:list, real_track:list, dt:float=0.1, u:float=2.0, sigma_real:float=0.25, sigma_t:float=1.2, should_graph:bool=False):
    # create KalmanFilter object
    kf = KalmanFilter(dt, u, sigma_real, sigma_t)

    predictions = []
    measurements = []
    for x in real_track:
        # Measurement
        z = kf.C * x + np.random.normal(0, 50)

        measurements.append(z.item(0))
        predictions.append(kf.predict()[0])
        kf.update(z.item(0))

    if should_graph:
        fig = plt.figure()
        fig.suptitle('Example of Kalman filter', fontsize=20)
        plt.plot(t, measurements, label='Measurements', color='b', linewidth=0.5)
        plt.plot(t, np.array(real_track), label='Real Track', color='y', linewidth=1.5)
        plt.plot(t, np.squeeze(predictions), label='Kalman Filter Prediction', color='r', linewidth=1.5)
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Position (m)', fontsize=20)
        plt.legend()
