#include <vector>
#include "Perceptron.h"

using namespace std;

class Layer {
public:
	Layer(unsigned perceptron_count, unsigned next_layer_perceptron_count, unsigned activation_function_code);
	vector<Perceptron> perceptrons;
};
