import subprocess
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-w', '--worker', type=int, default=1)
    parser.add_argument('-p', '--period', type=float, default=0)
    args = parser.parse_args()

    threads = []
    for i in range(args.worker):
        threads.append(subprocess.Popen(['python', 'client.py', '-a', args.abr, '-p', str(args.period)]))
    for thread in threads:
        thread.wait()
