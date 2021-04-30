#include <iostream>
#include <cstdlib>
#include <cmath>
#include "Perceptron.h"

Perceptron::Perceptron(unsigned next_layer_perceptron_count, unsigned perceptron_index, unsigned activation_function_code) 
	: perceptron_index(perceptron_index), 
	activation_function_code(activation_function_code) {
	for (size_t link_index = 0; link_index < next_layer_perceptron_count; link_index++) {
		this->output_weights.push_back(rand() / double(RAND_MAX));
	}
}

void Perceptron::set_output_value(double value) {
	this->output_value = value;
}

void Perceptron::feed_forward(const vector<Perceptron>& previous_layer_perceptrons) {
	double sum = 0.0;

	// Sum the previous layer's outputs
	size_t perceptron_count = previous_layer_perceptrons.size();
	for (unsigned perceptron_index = 0; perceptron_index < perceptron_count; perceptron_index++) {
		auto perceptron = previous_layer_perceptrons[perceptron_index];
		sum = sum + perceptron.get_output_value() * perceptron.output_weights[this->perceptron_index];
	}

	this->output_value = this->activation_function(sum);
}

void Perceptron::calculate_output_gradients(double expected_result) {
	double delta = expected_result - this->output_value;
	this->gradient = delta * this->activation_function_derivative(this->output_value);
}

void Perceptron::calculate_hidden_gradients(const vector<Perceptron>& next_layer_perceptrons) {
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed
	for (unsigned perceptron_index = 0; perceptron_index < next_layer_perceptrons.size(); perceptron_index++) {
		sum = sum + this->output_weights[perceptron_index] * next_layer_perceptrons[perceptron_index].gradient;
	}

	this->gradient = sum * this->activation_function_derivative(this->output_value);
}

double Perceptron::activation_function(double x) const {
	if (this->activation_function_code == 0) {
		return 1.0 / (1.0 + exp(-x));
	}
	else if (this->activation_function_code == 1) {
		return 0.1 * x;
	}
	else {
		return tanh(x);
	}
}

double Perceptron::activation_function_derivative(double y) const {
	if (this->activation_function_code == 0) {
		return y * (1.0 - y);
	}
	else if(this->activation_function_code == 1) {
		return 0.1;
	}
	else {
		return (1 - y * y);
	}
}

void Perceptron::update_output_weights(vector<Perceptron>& previous_layer_perceptrons) {
	for (unsigned perceptron_index = 0; perceptron_index < previous_layer_perceptrons.size(); perceptron_index++) {
		Perceptron& perceptron = previous_layer_perceptrons[perceptron_index];

		double delta_weight = Perceptron::learning_rate * perceptron.get_output_value() * this->gradient;
		perceptron.output_weights[this->perceptron_index] = perceptron.output_weights[this->perceptron_index] + delta_weight;
	}
}

double Perceptron::learning_rate = 0.0;