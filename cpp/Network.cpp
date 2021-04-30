#include <iostream>
#include <cassert>
#include "Network.h"

Network::Network(const vector<unsigned>& topology, unsigned activation_function_code) {
	for (size_t index = 0; index < topology.size(); index++) {
		auto perceptron_count = topology[index];
		auto next_layer_perceptron_count = index == (topology.size() - 1) ? int(0) : topology[index + 1];
		Layer layer(perceptron_count, next_layer_perceptron_count, activation_function_code);
		this->layers.push_back(layer);
	}
}

void Network::feed_forward(const vector<double>& input_values) {
	for (unsigned i = 0; i < input_values.size(); i++) {
		this->layers[0].perceptrons[i].set_output_value(input_values[i]);
	}

	for (unsigned layer_index = 1; layer_index < this->layers.size(); layer_index++) {
		for (unsigned perceptron_index = 0; perceptron_index < this->layers[layer_index].perceptrons.size(); perceptron_index++) {
			this->layers[layer_index].perceptrons[perceptron_index].feed_forward(this->layers[layer_index - 1].perceptrons);
		}
	}
}

void Network::back_propagate(const vector<double>& expected_results) {
	//Calculate output layer gradients
	Layer& output_layer = this->layers.back();
	for (unsigned perceptron_index = 0; perceptron_index < output_layer.perceptrons.size(); perceptron_index++) {
		output_layer.perceptrons[perceptron_index].calculate_output_gradients(expected_results[perceptron_index]);
	}

	//Calculate gradients on hidden layer
	for (unsigned layer_index = this->layers.size() - 2; layer_index > 0; layer_index--) {
		Layer& hidden_layer = this->layers[layer_index];
		Layer& next_layer = this->layers[layer_index + 1];

		for (unsigned perceptron_index = 0; perceptron_index < hidden_layer.perceptrons.size(); perceptron_index++) {
			hidden_layer.perceptrons[perceptron_index].calculate_hidden_gradients(next_layer.perceptrons);
		}
	}

	//Update the weights of all layers including input, hiddens, output layer
	for (unsigned layer_index = this->layers.size() - 1; layer_index > 0; layer_index--) {
		Layer& current_layer = this->layers[layer_index];
		Layer& previous_layer = this->layers[layer_index - 1];

		for (unsigned perceptron_index = 0; perceptron_index < current_layer.perceptrons.size(); perceptron_index++) {
			current_layer.perceptrons[perceptron_index].update_output_weights(previous_layer.perceptrons);
		}
	}
}

vector<double> Network::get_results() const {
	vector<double> results;

	auto output_perceptrons = this->layers.back().perceptrons;
	for (unsigned perceptron_index = 0; perceptron_index < output_perceptrons.size(); perceptron_index++) {
		results.push_back(output_perceptrons[perceptron_index].get_output_value());
	}

	return results;
}