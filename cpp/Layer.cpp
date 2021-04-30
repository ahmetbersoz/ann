#include <iostream>
#include "Layer.h"

Layer::Layer(unsigned perceptron_count, unsigned next_layer_perceptron_count, unsigned activation_function_code) {
	for (size_t index = 0; index < perceptron_count; index++) {
		Perceptron perceptron(next_layer_perceptron_count, index, activation_function_code);
		this->perceptrons.push_back(perceptron);
	}
}