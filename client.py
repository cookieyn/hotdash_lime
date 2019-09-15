import requests
import json
import argparse
import time

IP_ADDR = 'http://101.6.30.171:9999'


def sender(args):
    while True:
        t1 = time.time()
        time.sleep(args.period)
        env_post_data = {}
        env_post_data['last_bit_rate'] = 1
        env_post_data['buffer_size'] = 4.0
        env_post_data['rebuf'] = 2.7
        env_post_data['video_chunk_size'] = 450283
        env_post_data['delay'] = 2772
        env_post_data['video_chunk_remain'] = 47
        env_post_data['next_video_chunk_sizes'] = [155580, 398865, 611087, 957685, 1431809, 2123065]
        if args.abr == 'pensieve':
            pass
        elif args.abr == 'robustmpc':
            pass
        elif args.abr == 'hotdash':
            env_post_data['hotspot_chunks_remain'] = 5
            env_post_data['last_hotspot_bit_rate'] = 1
            env_post_data['next_hotspot_chunk_sizes'] = [150710, 374758, 584846, 852833, 1348011, 2055469]
            env_post_data['dist_from_hotspot_chunks'] = [0, 9, 17, 25, 38]
        else:
            raise NotImplementedError
        abr_request = requests.post(IP_ADDR, data=json.dumps(env_post_data))
        if abr_request.status_code == 200:
            bitrate_decision = abr_request.json()["bitrate"]
            print("Bitrate: " + str(bitrate_decision))
        else:
            bitrate_decision = -1
            print("Status code: " + abr_request.status_code)
        t2 = time.time()
        print(t2 - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-p', '--period', type=float, default=0)
    args = parser.parse_args()
    sender(args)
