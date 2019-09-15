import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.cluster import KMeans
import pickle as pk
import pydotplus
import pensieve
import pensiedt
import robustmpc
import robustmdt
import robustlin
import argparse
import load_trace
import fixed_env as env
from multiprocessing.dummy import Pool as ThreadPool
import limes
import pensilin
#from actor_wrapper import Actor_LIME
#NN_MODEL = './models/pretrain_linear_reward.ckpt'
import time

S_INFO = 6
S_INFO_P = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO_R = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_pensieve'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/pretrain_linear_reward.ckpt'
THREAD = 16


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


def get_rollouts(env, policy, args, n_batch_rollouts, lin=None):
    rollouts = []
    if lin is None:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env))
    else:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env, lime = lin))
    return rollouts


def resample(states, actions, serials, max_pts):
    idx = np.random.choice(len(states), size=max_pts)
    #print((serials[idx])[0],(serials[idx])[1],(serials[idx])[2])
    return states[idx], actions[idx], serials[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'bolaopt'])
    #parser.add_argument('-n', '--leaf-nodes', type=int)
    parser.add_argument('-q', '--qoe-metric', choices=['lin', 'log', 'hd'])
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-i', '--lin', action='store_true')
    parser.add_argument('-m', '--iters', type=int)
    parser.add_argument('-t', '--traces', choices=['norway', 'fcc', 'oboe'])    
    
    
    args = parser.parse_args()
    n_batch_rollouts = 10
    max_iters = args.iters
    max_pts = 20000
    train_frac = 0.8
    np.random.seed(RANDOM_SEED)
    states, actions, serials = [], [], []
    precision = []
    #trees = []
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names)

    if args.abr == 'pensieve':
        teacher = pensieve.Pensieve()
        student = pensilin.Pensilin()
        #test = pensieve.Pensieve()
    elif args.abr == 'robustmpc':
        teacher = robustmpc.RobustMPC()
        student = robustlin.Robustlin()
    else:
        raise NotImplementedError

    # Step 1: Initialization for the first iteration
    trace = get_rollouts(env=net_env, policy=teacher, args=args, n_batch_rollouts=n_batch_rollouts)
    states.extend((state for state, _, _ in trace))
    actions.extend((action for _, action, _ in trace))
    serials.extend(serial for _, _, serial in trace)
    
    
    def predict_fn2(data):
        if args.abr == 'pensieve':
            #print (data.shape)
            #print (data)
            input_data = np.zeros((data.shape[0], S_INFO, S_LEN))
            input_data[:, 0, -1] = data[:, 0]
            input_data[:, 1, -1] = data[:, 1]
            input_data[:, 2, :] = data[:, 2:10]
            input_data[:, 3, :] = data[:, 10:18]
            input_data[:, 4, :A_DIM] = data[:, 18:24]
            input_data[:, 5, -1] = data[:, 24]
            return teacher.actor.predict(input_data)
        else:
            pass
    for j in range(30,51):
        tt = open('dd.txt','a+')
        tt.write(str(j))
        tt.write('\n')
        tt.close()
        for i in range(max_iters):
            # Step 2:
            ff = open('dd.txt','a+')
            print('cluster number:{},Iteration {}/{}'.format(j, i, max_iters))
            cur_states, cur_actions, cur_serials = resample(np.array(states), np.array(actions), np.array(serials), max_pts)
            print('Training student with {} points'.format(len(cur_serials)))
            serials_train, actions_train, serials_val, actions_val = split_train_test(cur_serials, cur_actions, train_frac)
            #print(serials_train[8])
            #print(serials_train[9])
            #print(serials_train[10])
            #print(serials_train[11])
            #print(serials_train[12])
            #print(serials_train.shape[0])
            def predict_fn1(state,num):
                #print(state.shape)
                #print(state.shape[0])
                result = np.zeros((state.shape[0],6))
                #print(state.shape)
                #print(state_train.shape)
                if (num<10):
                    for i in range (num):
                        for j in range(6):
                            if(actions_train[i] == j):
                                for k in range(10):
                                    result[i+k*num][j] = 0.5
                            else:
                                for k in range(10):
                                    result[i+k*num][j] = 0.1
                else:
                    for i in range (num):
                    #print( state[i].shape)
                        for j in range(6):
                        #inx = np.where(serials_train == state[i])[0][0]
                            if (actions_train[i] == j):
                                result[i][j] = 0.5
                            else:
                                result[i][j] = 0.1
                    #print(i)
                #print(result)
                return result
            #time1 = time.clock()
            #print(time1)
            lin = limes.LIMESimpleModel(cluster_num = j, cluster_method = KMeans, random_state = None)
            lin.fit(serials_train, actions_train, predict_fn = predict_fn2, labels_num = 6)
            #time2 = time.clock()
            #times1=time2-time1
            #print(times1)
            #print('Train accuracy: {}'.format(np.mean(actions_train == lin.predict(serials_train))))
            #print('Val accuracy: {}'.format(np.mean(actions_val == lin.predict(serials_val))))
            precision.append(np.mean(lin.predict(serials_val) == actions_val))
            print('unpruned precision', precision[-1])
            ff.write(str(precision[-1]))
            ff.write('\n')
            student_trace = get_rollouts(env=net_env, policy=student, args=args, n_batch_rollouts=n_batch_rollouts,
                                     lin= lin)
            student_states = [state for state, _, _ in student_trace]
            student_actions = [action for _, action, _ in student_trace]
            student_serials = [serial for _, _, serial in student_trace]
            #time3 = time.clock()
            #times2=time3-time2
            #print(times2)
            if args.abr == 'pensieve':
                teacher_actions = map(teacher.predict, student_states)
            else:
                pool = ThreadPool(THREAD)
                teacher_actions = pool.map(teacher.predict, student_states)
                pool.close()
                pool.join()
            #time4 = time.clock()
            #times3=time4-time3
            #print(times3)
                #print (teacher_actions)
        # teacher_actions = []
        # for student_state in student_states:
        #     teacher_actions.append(teacher.predict(student_state))

            states.extend(student_states)
            actions.extend(teacher_actions)
            serials.extend(student_serials)
            ff.close()
        #print('cluster number ={}, average curancy: {}',format(j,sum(grade)/max_iters))
        #trees.append(lin)

    #best_lin = trees[-1]
    # save decision tree to file
    #with open('lime/' + args.abr + '.pk3', 'wb') as f:
        #pk.dump(lin, f)

    #dot_data = StringIO()
    #export_graphviz(lin, out_file=dot_data, filled=True)
    #out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #out_graph.write_svg('lime/' + args.abr + '.svg')


