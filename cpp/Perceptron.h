#include <vector>

using namespace std;

class Perceptron {
public:
	Perceptron(unsigned next_layer_perceptron_count, unsigned p_index, unsigned activation_function_code);
	void set_output_value(double value);
	double get_output_value() const { return this->output_value; }
	void feed_forward(const vector<Perceptron>& previous_layer_perceptrons);
	void calculate_output_gradients(double expected_result);
	void calculate_hidden_gradients(const vector<Perceptron>& next_layer_perceptrons);
	void update_output_weights(vector<Perceptron>& previous_layer_perceptrons);

	static double learning_rate;

private:
	unsigned activation_function_code;
	double activation_function(double x) const;
	double activation_function_derivative(double x) const;
	double output_value;
	vector<double> output_weights;
	vector<double> delta_weights;
	unsigned perceptron_index;
	double gradient;
};
