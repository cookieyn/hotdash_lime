import numpy as np
from sklearn.linear_model import LinearRegression

EPSILON = 1e-7


class MixtureLinearRegressionModel:
    def __init__(self, component, max_iteration=500):
        self.max_iteration = max_iteration
        self.component = component
        self.linear_models = []
        self.pi_ = None
        self.sigma_ = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.numSample = X.shape[0]
        self.numFeature = X.shape[1]

        self.__initInteration()
        for i in range(self.max_iteration):
            if not self.__E_step():
                self.__M_step()
            else:
                break

    def __E_step(self):
        linear_model = np.dot(self.X, self.coef_) + self.intercept_
        linear_offset = self.y.reshape(-1, 1) - linear_model
        probability = -0.5 * np.log(self.sigma_ + EPSILON) - (linear_offset**2) / (
            2 * self.sigma_ + EPSILON) + np.log(self.pi_ + EPSILON)

        updated_cluster_result = np.argmax(probability, axis=1)
        self.local_prediction = linear_model[np.arange(
            0, linear_model.shape[0]), updated_cluster_result]
        diff_cluster = np.sum(
            np.abs(updated_cluster_result - self.cluster_result))
        if diff_cluster == 0:
            return True
        else:
            self.cluster_result = updated_cluster_result
            return False

    def __M_step(self):
        for i in range(self.component):
            linear_reg = LinearRegression()
            inds = np.where(self.cluster_result == i)
            if len(inds[0]) != 0:
                linear_reg.fit(self.X[inds], self.y[inds])
                self.pi_[:, i] = np.sum(
                    np.int32(self.cluster_result == i)) / self.numSample
                self.coef_[:, i] = linear_reg.coef_
                self.intercept_[:, i] = linear_reg.intercept_
                self.sigma_[:, i] = np.mean(
                    (self.y[inds] - linear_reg.predict(self.X[inds]))**2)

    def __initInteration(self):
        assert self.X is not None

        self.pi_ = np.zeros((1, self.component), np.float32)
        self.sigma_ = np.zeros((1, self.component), np.float32)
        self.coef_ = np.zeros((self.numFeature, self.component), np.float32)
        self.intercept_ = np.zeros_like(self.sigma_)
        self.cluster_result = np.zeros(self.numSample)
        randDist = np.arange(0, self.X.shape[0])
        np.random.shuffle(randDist)
        for i in range(self.component):
            linear_reg = LinearRegression()
            pointRange = randDist[int(i * self.numSample / self.component):int(
                (i + 1) * self.numSample / self.component)]
            self.cluster_result[pointRange] = i
            linear_reg.fit(self.X[pointRange], self.y[pointRange])

            self.pi_[:, i] = len(pointRange) / self.numSample
            self.sigma_[:, i] = np.mean(
                (self.y[pointRange] - linear_reg.predict(self.X[pointRange]))
                ** 2)
            self.coef_[:, i] = linear_reg.coef_
            self.intercept_[:, i] = linear_reg.intercept_

    def predict(self, row):
        linear_model = np.dot(row, self.coef_) + self.intercept_
        return np.sum(self.pi_ * linear_model, axis=1)

    def local_prediction(self):
        return self.local_prediction
