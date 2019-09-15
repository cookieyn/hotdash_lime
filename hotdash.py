import os
import numpy as np
import tensorflow as tf
import fixed_env_hotdash as env
import multiprocessing as mp
import a3c_hotdash
import a3c
import load_trace
import get_reward

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
NN_MODEL_bitr = './models/pretrain_linear_reward.ckpt'

ACTIONS = [0, 1]


class Hotdash:
    def __init__(self):
        self.sess = tf.Session()
        self.actor = a3c_hotdash.ActorNetwork(self.sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM_prefetch,
                                              learning_rate=ACTOR_LR_RATE)
        self.critic = a3c_hotdash.CriticNetwork(self.sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(self.sess, NN_MODEL)
            print("Testing model 1 restored.")
        
        # reuse = True
        tf.reset_default_graph()
        
        self.sess_bitr = tf.Session()
        self.actor_bitr = a3c.ActorNetwork(self.sess_bitr, state_dim=[S_INFO_PENSIEVE, S_LEN], action_dim=A_DIM,
                                           learning_rate=ACTOR_LR_RATE)
        self.critic_bitr = a3c.CriticNetwork(self.sess_bitr, state_dim=[S_INFO_PENSIEVE, S_LEN],
                                             learning_rate=CRITIC_LR_RATE)

        self.sess_bitr.run(tf.global_variables_initializer())
        saver_bitr = tf.train.Saver()
        
        # restore neural net parameters
        if NN_MODEL_bitr is not None:  # NN_MODEL is the path to file
            saver_bitr.restore(self.sess_bitr, NN_MODEL_bitr)
            print("Testing model 2 restored.")
            
    def main(self, args, net_env=None, flag = 1):

        np.random.seed(RANDOM_SEED)
        viper_flag = True  # ?
        assert len(VIDEO_BIT_RATE) == A_DIM

        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)

        if net_env is None:
            viper_flag = False
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
            net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                                      all_file_names=all_file_names)
                                      
        if not viper_flag and args.log:
            log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
            log_file = open(log_path, 'wb')

        time_stamp = 0
        
        prefetch_decision = DEFAULT_PREFETCH
        next_normal_bitrate = DEFAULT_QUALITY
        next_hotspot_bitrate = DEFAULT_QUALITY
        bit_rate = 1
        action_vec = np.zeros(A_DIM_prefetch)
        action_vec[prefetch_decision] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        s_batch_pensieve1 = [np.zeros((S_INFO_PENSIEVE,S_LEN))]
        s_batch_pensieve2 = [np.zeros((S_INFO_PENSIEVE,S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        rollout = []
        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state_data_for_action = net_env.execute_action(prefetch_decision, next_normal_bitrate, next_hotspot_bitrate)
            
            # normal chunk state information
            delay = state_data_for_action['delay']
            sleep_time = state_data_for_action['sleep_time']
            last_bit_rate = state_data_for_action['last_bit_rate']
            play_buffer_size = state_data_for_action['play_buffer_size']
            rebuf = state_data_for_action['rebuf']
            video_chunk_size = state_data_for_action['video_chunk_size']
            next_video_chunk_sizes = state_data_for_action['next_video_chunk_sizes']
            end_of_video = state_data_for_action['end_of_video']
            video_chunk_remain = state_data_for_action['video_chunk_remain']
            current_seq_no = state_data_for_action['current_seq_no']
            log_prefetch_decision = state_data_for_action['log_prefetch_decision']

            # hotspot chunk state information
            was_hotspot_chunk = int(state_data_for_action['was_hotspot_chunk'])
            hotspot_chunks_remain = state_data_for_action['hotspot_chunks_remain']
            chunks_till_played = state_data_for_action['chunks_till_played']
            total_buffer_size = state_data_for_action['total_buffer_size']
            last_hotspot_bit_rate = state_data_for_action['last_hotspot_bit_rate']
            next_hotspot_chunk_sizes = state_data_for_action['next_hotspot_chunk_sizes']
            dist_from_hotspot_chunks = state_data_for_action['dist_from_hotspot_chunks']
            smoothness_eval_bitrates = state_data_for_action['smoothness_eval_bitrates']

            # abr decision state information
            normal_bitrate_pensieve = state_data_for_action['normal_bitrate_pensieve']
            hotspot_bitrate_pensieve = state_data_for_action['hotspot_bitrate_pensieve']

            last_overall_bitrate = last_bit_rate
            if prefetch_decision == 1:
                last_overall_bitrate = last_hotspot_bit_rate
            
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # calculating the reward, which is video quality - rebuffer penalty - smoothness

            if args.qoe_metric == 'lin':
                util_array = [util / M_IN_K for util in VIDEO_BIT_RATE]
            elif args.qoe_metric == 'log':
                util_array = [np.log(util / VIDEO_BIT_RATE[-1]) for util in VIDEO_BIT_RATE]
            elif args.qoe_metric == 'hd':
                util_array = HD_REWARD
            else:
                raise NotImplementedError
            reward_br = util_array[int(last_hotspot_bit_rate) if was_hotspot_chunk else int(last_bit_rate)]
            reward_rebuffering = REBUF_PENALTY * rebuf * 1.0
            reward_smoothness = 0.0
            if len(smoothness_eval_bitrates) > 1:
                for i in range(len(smoothness_eval_bitrates)-1):
                    reward_smoothness += 1.0 * SMOOTH_PENALTY * (1.0 * np.abs(VIDEO_BIT_RATE[int(
                        smoothness_eval_bitrates[i+1])] - VIDEO_BIT_RATE[int(smoothness_eval_bitrates[i])]) / M_IN_K)

            reward = (1.0*reward_br) - (1.0*reward_rebuffering) - (1.0*reward_smoothness)
            r_batch.append(reward)
            
            if args.log:
                # [time_ms, bit_rate, buff, volume, time, reward]
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(bytes(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[last_overall_bitrate]) + '\t' +
                               str(play_buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\t' +
                               str(log_prefetch_decision) + '\t' +
                               str(int(was_hotspot_chunk)) + '\t' +
                               str(current_seq_no) + '\n', encoding='utf-8'))
                log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            # Normal state S_ABR_INFO
            state[0, -1] = VIDEO_BIT_RATE[last_overall_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = play_buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            # Hotspot state S_HOT_INFO
            state[6, -1] = np.minimum(hotspot_chunks_remain, NUM_HOTSPOT_CHUNKS) / float(NUM_HOTSPOT_CHUNKS)
            state[7, -1] = np.minimum(chunks_till_played, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            state[8, -1] = total_buffer_size / BUFFER_NORM_FACTOR
            state[9, -1] = last_hotspot_bit_rate / float(np.max(VIDEO_BIT_RATE))
            state[10, :BITRATE_LEVELS] = np.array(next_hotspot_chunk_sizes) / M_IN_K / M_IN_K
            state[11, :NUM_HOTSPOT_CHUNKS] = (np.array(dist_from_hotspot_chunks) + CHUNK_TIL_VIDEO_END_CAP) / 2 / CHUNK_TIL_VIDEO_END_CAP

            # Bitrate actions state S_BRT_INFO
            state[12, -1] = normal_bitrate_pensieve / float(np.max(VIDEO_BIT_RATE))
            state[13, -1] = hotspot_bitrate_pensieve / float(np.max(VIDEO_BIT_RATE))
            
            if len(s_batch_pensieve1) == 0:
                state_info_pensieve_n = [np.zeros((S_INFO_PENSIEVE, S_LEN))]
            else:
                state_info_pensieve_n = np.array(s_batch_pensieve1[-1], copy=True)
            
            state_info_pensieve_n = np.roll(state_info_pensieve_n, -1, axis=1)
            state_info_pensieve_n[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state_info_pensieve_n[1, -1] = play_buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_n[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state_info_pensieve_n[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_n[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state_info_pensieve_n[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
            
            if len(s_batch_pensieve2) == 0:
                state_info_pensieve_h = [np.zeros((S_INFO_PENSIEVE, S_LEN))]
            else:
                state_info_pensieve_h = np.array(s_batch_pensieve2[-1], copy=True)
            
            state_info_pensieve_h = np.roll(state_info_pensieve_h, -1, axis=1)
            state_info_pensieve_h[0, -1] = VIDEO_BIT_RATE[last_bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state_info_pensieve_h[1, -1] = play_buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_h[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state_info_pensieve_h[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state_info_pensieve_h[4, :BITRATE_LEVELS] = np.array(next_hotspot_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state_info_pensieve_h[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            states_list = np.array([state, state_info_pensieve_n, state_info_pensieve_h])
            
            # --------------------------------------------------------
            
            all_act = self.predict(states_list)
            
            prefetch_decision = all_act[0]
            next_normal_bitrate = all_act[1]
            next_hotspot_bitrate = all_act[2]
            
            # if play_buffer_size < 500:
            #     prefetch_decision = 0
            # print("all act",all_act)
            
            s_batch.append(state)
            s_batch_pensieve1.append(state_info_pensieve_n)
            s_batch_pensieve2.append(state_info_pensieve_h)

            serialized_state = []
            # Log input of neural network
            serialized_state.append(state[0, -1])
            serialized_state.append(state[1, -1])
            for i in range(S_LEN):
                serialized_state.append(state[2, i])
            for i in range(S_LEN):
                serialized_state.append(state[3, i])
            for i in range(A_DIM):
                serialized_state.append(state[4, i])
            serialized_state.append(state[5, -1])
            for i in range(S_LEN):
                serialized_state.append(state[6, i])
            serialized_state.append(state[7, -1])
            serialized_state.append(state[8, -1])
            serialized_state.append(state[9, -1])
            for i in range(BITRATE_LEVELS):
                serialized_state.append(state[10, i])
            for i in range(NUM_HOTSPOT_CHUNKS):
                serialized_state.append(state[11, i])
            serialized_state.append(state[12, -1])
            serialized_state.append(state[13, -1])
            
            all_decision = [prefetch_decision, next_normal_bitrate, next_hotspot_bitrate]
            # print("decision:",all_decision)
            if flag == 0:
                rollout.append((states_list, all_decision, serialized_state))
            if flag == 1:
                serialized_state_n = serial(state_info_pensieve_n)
                rollout.append((state_info_pensieve_n, next_normal_bitrate, serialized_state_n))
            if end_of_video:
                if args.log:
                    log_file.write(bytes('\n', encoding='utf-8'))
                    log_file.close()
                    print("video count", video_count)

                prefetch_decision = DEFAULT_PREFETCH

                del s_batch[:]
                del s_batch_pensieve1[:]
                del s_batch_pensieve2[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[prefetch_decision] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                s_batch_pensieve1.append(np.zeros((S_INFO_PENSIEVE, S_LEN)))
                s_batch_pensieve2.append(np.zeros((S_INFO_PENSIEVE, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                if viper_flag:
                    break
                else:
                    video_count += 1
                    if video_count >= len(net_env.all_file_names):
                        break
                    if args.log:
                        log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx] + '_' + args.qoe_metric
                        log_file = open(log_path, 'wb')

        return rollout

    def get_abr_rl_bitrate(self, state_data):
        action_prob = self.actor_bitr.predict(np.reshape(state_data, (1, S_INFO_PENSIEVE, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        bitrate_decision = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

        return bitrate_decision

    def predict_prefetch(self, states):
        action_prob = self.actor.predict(np.reshape(states, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        prefetch_decision = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        
        return prefetch_decision

    def predict(self, states_list):
        prefetch_decision = self.predict_prefetch(states_list[0])
        next_normal_bitrate = self.get_abr_rl_bitrate(states_list[1])
        next_hotspot_bitrate = self.get_abr_rl_bitrate(states_list[2])
        
        return [prefetch_decision, next_normal_bitrate, next_hotspot_bitrate]


def serial(state):
    serialized_state = []
    serialized_state.append(state[0, -1])
    serialized_state.append(state[1, -1])
    for i in range(S_LEN):
        serialized_state.append(state[2, i])
    for i in range(S_LEN):
        serialized_state.append(state[3, i])
    for i in range(A_DIM):
        serialized_state.append(state[4, i])
    serialized_state.append(state[5, -1])
    return serialized_state
