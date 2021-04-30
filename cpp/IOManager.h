#include <string>
#include <vector>

using namespace std;

class IOManager {
public:
	IOManager();

	void read_input(const string& filename);
	void read_dataset();
	string get_dataset_filename() const { return this->dataset_filename; }
	string get_output_filename() const { return "output_" + this->dataset_filename; }
	int get_topology_count() const { return this->topology_count; }
	vector<unsigned> get_topology(int topology_index) const { return this->topologies[topology_index]; }
	vector<vector<double>> get_input_values() const { return this->input_values; }
	vector<vector<double>> get_expected_outputs() const { return this->expected_outputs; }
	vector<unsigned> get_activation_function_codes() const { return this->activation_function_codes; }
	vector<double> get_learning_rates() const { return this->learning_rates; }
	int get_number_of_epochs() const { return this->number_of_epochs; }

private:
	int topology_count;
	vector<vector<unsigned>> topologies; // {first topology, second topology, ...}
	string dataset_filename;
	int number_of_features;
	vector<unsigned> activation_function_codes; // 0 for sigmoid, 1 linear. {activation_function_code_for_topology_1, ...}
	vector<double> learning_rates;
	int number_of_epochs;
	vector<vector<double>> input_values; // {{feature 1, feature 2, ...}, {feature 1, feature 2, ...}, ...}
	vector<vector<double>> expected_outputs; // {expected outputs of row 1, expected outputs of row 2, ...}

	vector<unsigned> split_line_to_unsigned(const string& line) const;
	vector<double> split_line_to_double(const string& line) const;
};