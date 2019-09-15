import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import time


class LIMESimpleModel:
    def __init__(self, cluster_num, cluster_method=KMeans, random_state=None):
        self.random_state = check_random_state(random_state)
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method(
            n_clusters=cluster_num, random_state=self.random_state)
        self.models = []

    def fit(self, X, y, predict_fn, labels_num):
        self.cluster_labels = self.cluster_method.fit_predict(X)
        #print(X.shape[1])
       
        for i in range(self.cluster_num):
            inds = np.where(self.cluster_labels == i)
            explainer = LimeTabularExplainer(
                X[inds],
                discretize_continuous=False,
                sample_around_instance=True)
            #print(np.squeeze(X[inds, :]))
            #print (self.cluster_method.cluster_centers_[i])
            #time1=time.clock()
            simplified_models = explainer.explain_instance(
                    self.cluster_method.cluster_centers_[i],
                    predict_fn,
                    num_samples=10000,
                    labels=range(labels_num),
                    num_features=X.shape[1],
                    retrive_model=True)
            #print(type(simplified_models))
            coef_ = np.zeros((X.shape[1], labels_num))
            intercept_ = np.zeros((1, labels_num))
            #time2=time.clock()
            #time3 = time2-time1
            #print("explain_instance")
            #print(time3)
            for idx in range(labels_num):
                coef_[:, idx] = simplified_models[idx].coef_
                intercept_[0, idx] = simplified_models[idx].intercept_

            self.models.append((coef_, intercept_))
            #print (self.models)

    def predict(self, x):

        cluster_result = self.cluster_method.predict(x)
        prediction_result = np.zeros(x.shape[0])

        for i in range(self.cluster_num):
            inds = np.where(cluster_result == i)
            if not len(inds[0]):
                continue
            predict_values = np.dot(np.squeeze(x[inds, :]),
                                    self.models[i][0]) + self.models[i][1]
            prediction_result[inds] = np.argmax(predict_values, axis=1)
        #print (prediction_result.shape)
        #print (prediction_result)
        return prediction_result


if __name__ == "__main__":
    from actor_wrapper import Actor_LIME
    from utils import collect_dataset
    from sklearn.model_selection import train_test_split

    actor = Actor_LIME(
        nn_model="../pensieve_test/models/pretrain_linear_reward.ckpt")

    dataset = collect_dataset().values
    labels = np.argmax(actor.predict(dataset), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    with open("./lime_extended_performance.csv", "w") as FILE:
        for j in range(100):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset, labels, test_size=0.2)
            for i in range(1, 50):
                lime_model = LIMESimpleModel(cluster_num=i)
                lime_model.fit(
                    X=X_train,
                    y=y_train,
                    predict_fn=actor.predict,
                    labels_num=6)
                train_err = np.sum(
                    np.int32(lime_model.predict(X_train) != y_train)
                ) / X_train.shape[0]
                test_err = np.sum(
                    np.int32(lime_model.predict(X_test) != y_test)
                ) / X_test.shape[0]

                train_rmse = np.sqrt(
                    np.mean((lime_model.predict(X_train) - y_train)**2))
                test_rmse = np.sqrt(
                    np.mean((lime_model.predict(X_test) - y_test)**2))

                FILE.write(f"{i},{1 - train_err},{1 - test_err},"
                           f"{train_rmse},{test_rmse}\n")

