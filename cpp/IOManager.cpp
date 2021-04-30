#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "IOManager.h"

using namespace std;

IOManager::IOManager() {

}

void IOManager::read_input(const string& filename) {
	string line;
	ifstream input_file(filename);

	if (input_file.is_open()) {
		string temp;

		getline(input_file, this->dataset_filename);
		
		getline(input_file, temp);
		this->number_of_features = stoi(temp);
		
		getline(input_file, temp);
		this->topology_count = stoi(temp);

		for (int index = 0; index < this->topology_count; index++) {
			vector<unsigned> topology;
			topology.push_back(this->number_of_features);

			getline(input_file, temp);
			auto hidden_layer_perceptron_count = stoi(temp);
			topology.push_back(hidden_layer_perceptron_count);

			topology.push_back(1);

			topologies.push_back(topology);
		}

		getline(input_file, temp);
		this->activation_function_codes = this->split_line_to_unsigned(temp);

		getline(input_file, temp);
		this->learning_rates = this->split_line_to_double(temp);

		getline(input_file, temp);
		this->number_of_epochs = stoi(temp);

		input_file.close();
	}
	else {
		cout << "Unable to open input file.";
	}
}

void IOManager::read_dataset() {
	string line;
	ifstream dataset_file(this->dataset_filename);

	if (dataset_file.is_open())
	{
		string temp;
		while (getline(dataset_file, temp)) {
			auto row_values = this->split_line_to_double(temp);
			this->expected_outputs.push_back({ row_values[row_values.size() - 1] });
			row_values.pop_back();
			this->input_values.push_back(row_values);
		}

		dataset_file.close();
	}
	else {
		cout << "Unable to open dataset file.";
	}
}

vector<unsigned> IOManager::split_line_to_unsigned(const string& line) const {
	vector<unsigned> splitted;
	
	istringstream is(line);
	int n;
	while (is >> n) {
		splitted.push_back(n);
	}

	return splitted;
}

vector<double> IOManager::split_line_to_double(const string& line) const {
	vector<double> splitted;

	istringstream is(line);
	double n;
	while (is >> n) {
		splitted.push_back(n);
	}

	return splitted;
}

