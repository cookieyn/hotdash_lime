import os
import numpy as np


trace_path = './sim_fcc/'


def main():
	files = os.listdir(trace_path)
	for trace_file in files:
		with open(trace_path + trace_file, 'rb') as f:
			time_ms = []
			bw_mbps = []
			for line in f:
				parse = line.split()
				time_ms.append(float(parse[0]))
				bw_mbps.append(float(parse[1]))
			Ex = 0
			Exx = 0
			for idx in range(1, len(time_ms)):
				Ex += 0.5 * (bw_mbps[idx-1] + bw_mbps[idx]) * (time_ms[idx] - time_ms[idx-1])
			Ex = Ex / time_ms[-1]
			for idx in range(1, len(time_ms)):
				Exx += 0.25 * (bw_mbps[idx-1] + bw_mbps[idx] - 2 * Ex) * (bw_mbps[idx-1] + bw_mbps[idx] - 2 * Ex) * (time_ms[idx] - time_ms[idx-1])
			Exx = Exx / time_ms[-1]
			Exx = np.sqrt(Exx)
			print(Ex, Exx)		


if __name__ == '__main__':
	main()
