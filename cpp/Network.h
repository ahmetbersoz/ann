#include <vector>
#include "Layer.h"

using namespace std;

class Network {
public:
	Network(const vector<unsigned>& topology, unsigned activation_function_code);

	void feed_forward(const vector<double>& input_values);
	void back_propagate(const vector<double>& expected_results);
	vector<double> get_results() const;

private:
	vector<Layer> layers; 
};