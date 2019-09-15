import os


SRC_DIR = './cooked_traces/'
DST_DIR = './sim_traces/'
UPDATE = 100
MS_IN_S = 1000.0
MTU = 1.5*8


def main():
    files = os.listdir(SRC_DIR)
    for f in files:
        print(f)
        src_path = SRC_DIR + f
        dst_path = DST_DIR + f
        with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
            last_t = 0
            mtu_count = 0
            for line in src:
                cur_t = int(line)
                if cur_t == 0:
                    continue
                mtu_count += 1
                if cur_t - last_t > UPDATE:
                    est_bw = mtu_count * MTU / (cur_t - last_t)
                    dst.write(bytes(str(cur_t / MS_IN_S) + '\t' + str(est_bw) + '\n', encoding='utf-8'))
                    mtu_count = 0
                    last_t = cur_t


if __name__ == '__main__':
    main()
