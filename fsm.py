import numpy as np
from get_chunk_size import get_chunk_size

HORIZON = 5
CHUNK_LEN = 4.0
BITRATE_NUM = 6

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

CHUNK_ON = 0
CHUNK_SWITCH = 1
OPTIMIZED = 2
M_IN_K = 1000.0


class Trajectory:
    def __init__(self, index, bitrate, buffer, last_bitrate, states, args):
        self.chunk_completed = 0
        self.chunk_remain = get_chunk_size(bitrate, index)
        self.proc_time = 0
        self.chunk_index = index
        self.chunk_init_bitrate = bitrate
        self.rebuf = []
        self.quality = [bitrate]
        self.buffer = buffer
        self.last_bitrate = [last_bitrate]
        self.trans_msg = None
        self.end = False
        self.states = states
        self.args = args

    def apply(self, chunk_size, delay):
        if self.end:
            return OPTIMIZED

        delay /= M_IN_K
        if chunk_size <= self.chunk_remain:
            self.chunk_remain -= chunk_size
            self.proc_time += delay
            return CHUNK_ON
        else:
            self.proc_time += delay * (self.chunk_remain / chunk_size)
            self.chunk_completed += 1
            self.rebuf.append(max(0.0, self.proc_time - self.buffer))

            if self.chunk_completed >= HORIZON:
                self.end = True
                return OPTIMIZED
            else:

                self.buffer = max(0, self.buffer - self.proc_time) + CHUNK_LEN
                self.trans_msg = chunk_size - self.chunk_remain, delay - delay * (self.chunk_remain / chunk_size)

                self.states = np.roll(self.states, -1, axis=1)
                if self.args.abr == 'pensieve':
                    self.states[0, -1] = VIDEO_BIT_RATE[self.quality[-1]] / float(np.max(VIDEO_BIT_RATE))
                    self.states[1, -1] = self.buffer / BUFFER_NORM_FACTOR  # 10 sec
                    self.states[2, -1] = float(get_chunk_size(self.quality[-1], self.chunk_index + self.chunk_completed - 1)) / float(self.proc_time) / M_IN_K  # kilo byte / ms
                    self.states[3, -1] = float(self.proc_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    self.states[4, :A_DIM] = np.array([get_chunk_size(r, self.chunk_index + self.chunk_completed)
                                                       for r in range(BITRATE_NUM)]) / M_IN_K / M_IN_K  # mega byte
                    self.states[5, -1] = self.states[5, -1] - 1/float(CHUNK_TIL_VIDEO_END_CAP)
                elif self.args.abr == 'robustmpc':
                    self.states[0, -1] = VIDEO_BIT_RATE[self.quality[-1]] / float(np.max(VIDEO_BIT_RATE))  # last quality
                    self.states[1, -1] = self.buffer / BUFFER_NORM_FACTOR
                    self.states[2, -1] = self.rebuf[-1]
                    self.states[3, -1] = float(self.proc_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    self.states[4, -1] = self.states[4, -1] - 1/float(CHUNK_TIL_VIDEO_END_CAP)

                return CHUNK_SWITCH

    def next_chunk(self, bitrate):
        if self.end:
            return OPTIMIZED

        self.last_bitrate.append(self.quality[-1])
        self.proc_time = 0
        self.quality.append(bitrate)
        self.chunk_remain = get_chunk_size(quality=bitrate, index=self.chunk_index + self.chunk_completed)


class FSM:
    def __init__(self, tree):
        self.tree = tree
        self.chunk_leaf = {}
        self.states = np.zeros(self.tree.tree_.node_count)

    def get_action(self, node_index):
        return self.tree.tree_.value[node_index][0].tolist().index(max(self.tree.tree_.value[node_index][0]))

    def update(self, node_index, action):
        if self.states[node_index] == 0:
            self.states[node_index] = action
        else:
            if self.states[node_index] == action:
                classes = np.zeros(BITRATE_NUM)
                classes[min(max(self.get_action(node_index) + action, 0), 5)] = 100
                self.tree.tree_.value[node_index][0] = classes
            self.states[node_index] = 0

    def predict(self,x):
        return self.tree.predict(x)
