// ffnnwbp.cpp : Defines the entry point for the console application.
//

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "stdafx.h"
#include "IOManager.h"
#include "Network.h"

using namespace std;

void print_status(double progress) {
	int barWidth = 70;

	cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) cout << "=";
		else if (i == pos) cout << ">";
		else cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " %\r";
	cout.flush();
}

void flush_status() {
	int barWidth = 100;

	for (int i = 0; i < barWidth; ++i) {
		cout << " ";
	}

	cout << "\r";
}

int _tmain(int argc, _TCHAR* argv[])
{
	string input_filename = "nn_energy_normalized.dat"; // CHANGE THIS LINE TO CHANGE INPUTS

	IOManager io_manager;

	io_manager.read_input(input_filename);
	io_manager.read_dataset();

	size_t data_count = io_manager.get_input_values().size();

	auto training_data_count = int(data_count * 0.6);
	auto validation_data_count = int(data_count * 0.2);
	auto test_data_count = int(data_count * 0.2);

	vector<vector<int>> training_folds(4);
	vector<vector<int>> testing_folds(4);

	for (int index = 0; index < (training_data_count + validation_data_count); index++) {
		if (index < validation_data_count) {
			testing_folds[0].push_back(index);
			training_folds[1].push_back(index);
			training_folds[2].push_back(index);
			training_folds[3].push_back(index);
		}
		else if (index >= validation_data_count && index < 2 * validation_data_count) {
			training_folds[0].push_back(index);
			testing_folds[1].push_back(index);
			training_folds[2].push_back(index);
			training_folds[3].push_back(index);
		}
		else if (index >= 2 * validation_data_count && index < 3 * validation_data_count) {
			training_folds[0].push_back(index);
			training_folds[1].push_back(index);
			testing_folds[2].push_back(index);
			training_folds[3].push_back(index);
		}
		else if (index >= 3 * validation_data_count && index < 4 * validation_data_count) {
			training_folds[0].push_back(index);
			training_folds[1].push_back(index);
			training_folds[2].push_back(index);
			testing_folds[3].push_back(index);
		}
	}

	vector<int> test_indexes;

	for (size_t index = (training_data_count + validation_data_count); index < data_count; index++) {
		test_indexes.push_back(index);
	}

	auto all_inputs = io_manager.get_input_values();
	auto all_outputs = io_manager.get_expected_outputs();

	cout << "Process started. You can find the output in file: " << io_manager.get_output_filename() << "\n";

	ofstream output_file;
	output_file.open(io_manager.get_output_filename());

	output_file << "GENERAL INFORMATION\n";
	output_file << "Dataset: " << io_manager.get_dataset_filename() << "\n";
	output_file << "Dataset entry count: " << io_manager.get_input_values().size() << "\n";
	output_file << "Cross validation technique: 4-fold\n";
	output_file << "Training + validation dataset entry count: " << (training_data_count + validation_data_count) << "\n";
	output_file << "Testing dataset entry count: " << test_data_count << "\n";
	output_file << "Number of epochs: " << io_manager.get_number_of_epochs() << "\n";
	output_file << "Activation function codes: 0 -> sigmoidal, 1 -> linear (y=0.1x)\n";
	output_file << "\n";

    double best_set_mae = 100000000000.0;
    double y1_range = 37.0;
	int best_topology_index;
	int best_activation_function_code;
	double best_learning_rate;
	for (int topology_index = 0; topology_index < io_manager.get_topology_count(); topology_index++) {
		cout << "Topology: " << (topology_index + 1) << "/" << io_manager.get_topology_count() << endl;

		auto topology = io_manager.get_topology(topology_index);

		for (size_t activation_index = 0; activation_index < io_manager.get_activation_function_codes().size(); activation_index++) {
			cout << "Activation function: " << (activation_index + 1) << "/" << io_manager.get_activation_function_codes().size() << endl;

			auto activation_function_code = io_manager.get_activation_function_codes()[activation_index];

			for (size_t learning_index = 0; learning_index < io_manager.get_learning_rates().size(); learning_index++) {
				cout << "Learning rate index: " << (learning_index + 1) << "/" << io_manager.get_learning_rates().size() << endl;

				Network network(topology, activation_function_code);
				Perceptron::learning_rate = io_manager.get_learning_rates()[learning_index];

				output_file << "CURRENT PARAMETERS\n";
				output_file << "Hidden layer perceptron count: " << io_manager.get_topology(topology_index)[1] << "\n";
				output_file << "Activation function: " << activation_function_code << "\n";
				output_file << "Learning rate: " << io_manager.get_learning_rates()[learning_index] << "\n";

				double parameter_set_mae = 0;
				for (int fold_index = 0; fold_index < 4; fold_index++) {
					cout << "Fold index: " << (fold_index + 1) << "/4" << endl;

					for (int epoch_index = 0; epoch_index < io_manager.get_number_of_epochs(); epoch_index++) {
						print_status(static_cast<double>(epoch_index + 1) / io_manager.get_number_of_epochs());

						for (int data_index = 0; data_index < validation_data_count; data_index++) {
							network.feed_forward(all_inputs[training_folds[fold_index][data_index]]);
							network.back_propagate(all_outputs[training_folds[fold_index][data_index]]);
						}
					}

					flush_status();

					double mae = 0;
					for (int data_index = 0; data_index < validation_data_count; data_index++) {
						network.feed_forward(all_inputs[testing_folds[fold_index][data_index]]);
						auto calculated_output = network.get_results()[0];
						auto expected_output = all_outputs[testing_folds[fold_index][data_index]][0];

                        mae += abs(calculated_output - expected_output);
					}
                    mae = mae / validation_data_count * y1_range;

                    output_file << "Fold index: " << fold_index << " ->  MAE = " << mae << "\n";

                    parameter_set_mae += mae;
				}
                output_file << "Parameter set MAE: " << parameter_set_mae << "\n\n";

                if (parameter_set_mae < best_set_mae) {
					best_topology_index = topology_index;
					best_activation_function_code = activation_function_code;
					best_learning_rate = io_manager.get_learning_rates()[learning_index];

                    best_set_mae = parameter_set_mae;
				}
			}
		}
	}

	output_file << "BEST PARAMETERS\n";
	output_file << "Hidden layer perceptron count: " << io_manager.get_topology(best_topology_index)[1] << "\n";
	output_file << "Activation function code: " << best_activation_function_code << "\n";
	output_file << "Learning rate: " << best_learning_rate << "\n\n";

	auto topology = io_manager.get_topology(best_topology_index);
	Network network(topology, best_activation_function_code);
	Perceptron::learning_rate = best_learning_rate;

	output_file << "TEST DATA\n";
	output_file << "Training network setup with best parameters.\n";

	vector<int> training_indexes;

	for (size_t index = 0; index < (training_data_count + validation_data_count); index++) {
		training_indexes.push_back(index);
	}

	for (int epoch_index = 0; epoch_index < io_manager.get_number_of_epochs(); epoch_index++) {
		for (int data_index = 0; data_index < (training_data_count + validation_data_count); data_index++) {
			network.feed_forward(all_inputs[training_indexes[data_index]]);
			network.back_propagate(all_outputs[training_indexes[data_index]]);
		}
	}

	double mae = 0;
	for (int data_index = 0; data_index < test_data_count; data_index++) {
		network.feed_forward(all_inputs[test_indexes[data_index]]);
		auto calculated_output = network.get_results()[0];
		auto expected_output = all_outputs[test_indexes[data_index]][0];

		output_file << "Calc:" << calculated_output << " Exp:" << expected_output << "\n";

        mae += abs(calculated_output - expected_output);
	}
    mae = mae / test_data_count * y1_range;

	output_file << "Running test data with trained network.\n";
    output_file << "Test Data MAE = " << mae << "\n";

	output_file.close();

	cout << "Done!";

	getchar();
	return 0;
}
