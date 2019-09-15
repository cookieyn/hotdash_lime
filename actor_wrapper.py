import tensorflow as tf
import sys
sys.path.append("../pensieve_test/")
import a3c  # noqa: E402
import numpy as np  # noqa: E402

S_INFO = 6
S_LEN = 8
A_DIM = 6


class Actor_LIME:
    def __init__(self, s_info=S_INFO, s_len=S_LEN, a_dim=A_DIM, nn_model=None):
        self.sess = tf.Session()

        self.actor = a3c.ActorNetwork(
            self.sess,
            state_dim=[s_info, s_len],
            action_dim=a_dim,
            learning_rate= 0.0001)

        saver = tf.train.Saver()  # save neural net parameters

        if nn_model is not None:  # nn_model is the path to file
            saver.restore(self.sess, nn_model)

    def predict(self, data):
        input_data = np.zeros((data.shape[0], S_INFO, S_LEN))
        input_data[:, 0, -1] = data[:, 0]
        input_data[:, 1, -1] = data[:, 1]
        input_data[:, 2, :] = data[:, 2:10]
        input_data[:, 3, :] = data[:, 10:18]
        input_data[:, 4, :A_DIM] = data[:, 18:24]
        input_data[:, 5, -1] = data[:, 24]

        return self.actor.predict(input_data)

    def __del__(self):
        self.sess.close()
