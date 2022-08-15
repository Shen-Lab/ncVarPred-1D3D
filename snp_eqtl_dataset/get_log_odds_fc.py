import numpy as np
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'deepsea tunning')
	parser.add_argument('--input_path', type = str)
	parser.add_argument('--experiment_name', type = str)
	parser.add_argument('--output_path', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	input_path = args.input_path
	experiment_name = args.experiment_name
	output_path = args.output_path
	wt_input = np.load(input_path + experiment_name + '_wt_prediction.npy')
	mt_input = np.load(input_path + experiment_name + '_mt_prediction.npy')
	threshold = 1e-12
	wt_input[wt_input < threshold] = threshold
	wt_input[wt_input > (1 - threshold)] = 1 - threshold
	mt_input[mt_input < threshold] = threshold
	mt_input[mt_input > (1 - threshold)] = 1 - threshold
	output = np.log2(wt_input/(1 - wt_input)) - np.log2(mt_input/(1 - mt_input))
	np.save(output_path + experiment_name + '_diff.npy', wt_input - mt_input)
	np.save(output_path + experiment_name + '_log_odds_fc.npy', output)

if __name__=='__main__':
	main()

