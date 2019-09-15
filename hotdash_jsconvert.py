import os
import numpy as np
import tensorflow as tf
import fixed_env_hotdash as env
import multiprocessing as mp
import a3c_hotdash
from tensorflow.python.tools.freeze_graph import freeze_graph

################################################################

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_ABR_INFO = 6
# next_hs_chunk_size, num_hs_remaining, num_chunks_remaining_till_hs_chunk_played, play_buffer_size,
# bitrate_last_hs, dist_vector_from_hs_chunks
S_HOT_INFO = 6
S_BRT_INFO = 2  # next_bit_rate, next_hs_bit_rate
S_INFO = S_ABR_INFO + S_HOT_INFO + S_BRT_INFO
S_INFO_PENSIEVE = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
A_DIM_prefetch =2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
# NUM_AGENTS = 1
S_INFO_bitr = 6


TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
ENTROPY_CHANGE_INTERVAL = 20000
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
NUM_HOTSPOT_CHUNKS = 5
M_IN_K = 1000.0
BITRATE_LEVELS = 6
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
DEFAULT_PREFETCH = 0 # default prefetch decision without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_hotdash'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/nn_model_ep_108000.ckpt'
# NN_MODEL_bitr = './models/pretrain_linear_reward.ckpt'
frozen_graph = './graphs/hotdash_2.pb'
ACTIONS = [0, 1]


def hotdash_jsconvert():
    sess = tf.Session()
    actor = a3c_hotdash.ActorNetwork(sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM_prefetch,
                                     learning_rate=ACTOR_LR_RATE)
    critic = a3c_hotdash.CriticNetwork(sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)

    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, NN_MODEL)
    print("Testing model 1 restored.")
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        ['add_1']
    )
    with tf.gfile.GFile(frozen_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    # reuse = True
    # tf.reset_default_graph()
    #
    # sess_bitr = tf.Session()
    # actor_bitr = a3c.ActorNetwork(sess_bitr, state_dim=[S_INFO_PENSIEVE, S_LEN], action_dim=A_DIM,
    #                                    learning_rate=ACTOR_LR_RATE)
    # critic_bitr = a3c.CriticNetwork(sess_bitr, state_dim=[S_INFO_PENSIEVE, S_LEN],
    #                                      learning_rate=CRITIC_LR_RATE)
    #
    # sess_bitr.run(tf.global_variables_initializer())
    # saver_bitr = tf.train.Saver()
    #
    # # restore neural net parameters
    # if NN_MODEL_bitr is not None:  # NN_MODEL is the path to file
    #     saver.restore(sess_bitr, NN_MODEL_bitr)
    #     print("Testing model 2 restored.")


if __name__ == '__main__':
    hotdash_jsconvert()
