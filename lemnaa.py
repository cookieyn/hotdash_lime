import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


class LEMNASimpleModel:
    def __init__(self, cluster_num, cluster_method=KMeans, random_state=None):
        self.random_state = check_random_state(random_state)
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method(
            n_clusters=cluster_num, random_state=self.random_state)
        self.models = []

    def fit(self, X, y, lemna_component, predict_fn, labels_num):
        self.cluster_labels = self.cluster_method.fit_predict(X)
        self.num_features = X.shape[1]

        for i in range(self.cluster_num):
            inds = np.where(self.cluster_labels == i)
            explainer = LimeTabularExplainer(
                np.squeeze(X[inds, :]),
                discretize_continuous=False,
                sample_around_instance=True)
                
            simplified_models = explainer.explain_instance_with_lemna(
                self.cluster_method.cluster_centers_[i],
                predict_fn,
                lemna_component=lemna_component,
                num_samples=5000,
                labels=range(labels_num),
                num_features=X.shape[1],
                retrive_model=True)

            # coef_ is a 3-d matrix feature_num * lemna_component * labels_num
            # intercept is a 2-d matrix lemna_component * labels_num
            coef_ = np.zeros((X.shape[1], lemna_component, labels_num))
            intercept_ = np.zeros((1, lemna_component, labels_num))

            for idx in range(labels_num):
                coef_[:, :, idx] = simplified_models[idx].coef_
                intercept_[0, :, idx] = simplified_models[idx].intercept_
                pi_ = simplified_models[idx].pi_

            self.models.append((coef_, intercept_, pi_))

    def predict(self, x):

        cluster_result = self.cluster_method.predict(x)
        prediction_result = np.zeros(x.shape[0])

        for i in range(self.cluster_num):
            inds = np.where(cluster_result == i)
            if not len(inds[0]):
                continue

            # here
            # feature_num * lemna_component * labels_num
            # -> lemna_component * feature_num * labels_num

            # sample_num * num_feature dot
            # lemna_component * feature_num * labels_num
            # -> sample_num * lemna_component * labels_num

            predict_values = np.dot(
                np.squeeze(x[inds, :]).reshape(-1, self.num_features),
                np.transpose(self.models[i][0], (1, 0, 2))) + self.models[i][1]

            # lemna_component dot
            # sample_num * lemna_component * labels_num
            # -> sample_num * label_num
            predict_values = np.dot(self.models[i][2].reshape(1, -1),
                                    predict_values)  # noqa: E501
            predict_values = np.squeeze(predict_values, axis=0)
            prediction_result[inds] = np.argmax(predict_values, axis=1)

        return prediction_result

