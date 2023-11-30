
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include<vector>
#include<cstdlib>
#include <cmath>




using namespace std;

class neuron {
public:
	double apredizaje;
	double salida;
	double expected;
	char type;
	double value;
	double error;
	double y = 0.0, y2 = 0.0;
	vector<neuron*> neuronas;
	vector<double> weights;
	double sum_error;
	int op = 1000;



	neuron(char t, double val, double apren) {
		type = t;
		value = val;
		apredizaje = apren;
	}

	void add(neuron* a) {
		neuronas.push_back(a);
	}

	double act1(double sum) {
		return 1.0 / (1.0 + exp(-sum));
	}

	double act1_derivative(double sum) {
		double sigmoid = 1.0 / (1.0 + exp(-sum));
		return sigmoid * (1.0 - sigmoid);
	}


	double act2(double sum) {
		return 2.0 / (1.0 + exp(-2.0 * sum)) - 1;
	}

	double derivative_act2(double sum) {
		double exp_term = exp(-2.0 * sum);
		double denominator = pow(1.0 + exp_term, 2.0);
		return 4.0 * exp_term / denominator;
	}


	void calculate_y() {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}
		y = act1(sum);
		value = y;
	}

	void calculate_y3() {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}
		y = act2(sum);
		value = y*op;
	}

	void calculate_y2() {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}
		y = act2(sum);
		value = y;
	}

	void calculate_e() {
		error = (value - y);
	}

	void new_weights(double apred,double err) {
		for (int i = 0; i < neuronas.size(); i++) {
			weights[i] += (neuronas[i]->value *apred)* act1_derivative(err) ;
		}
	}

	double expe() {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}
		expected = act1(sum);
		return sum;
	}


	void new_errors(double error1) {

		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}
	
		for (int i = 0; i < neuronas.size(); i++) {
			neuronas[i]->error = weights[i] * error1 * act1_derivative(sum);
		}
	}


	void new_errors2(double er) {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			sum += weights[i] * neuronas[i]->value;
		}

		for (int i = 0; i < neuronas.size(); i++) {
			neuronas[i]->error = act1_derivative(sum)*er;
		}
	}

	double sum_e() {
		double sum = 0.0;
		for (int i = 0; i < neuronas.size(); i++) {
			weights[i] += (weights[i] * neuronas[i]->value)*act1_derivative(error);
		}
	}


	void weights_hidden(double apred) {
		for (int i = 0; i < neuronas.size(); i++) {
			weights[i] += (neuronas[i]->value * apred)*error;
		}
	}


};



class perceptron {
public:
	vector<vector<double>> dataset;
	vector<double> entry;
	vector<double> exit;
	vector<vector<double>> testingset;
	vector<int> vals;


	vector<vector<double>> training;
	vector<vector<double>> check = {
						{ 1.0,0.0,0.0,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0 },
						{ 0.0,1.0,0.0,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0 },
						{ 0.0,0.0,1.0,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0 },
						{ 0.0,0.0,0.0,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0 },
						{ 0.0,0.0,0.0,0.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0 },

	};


	int entrada;
	int salida;
	int hidden1, hidden2, hidden3;

	vector<neuron*> n_input;

	vector<neuron*> H1;
	vector<neuron*> H2;
	vector<neuron*> H3;

	vector<neuron*> n_output;

	vector<double> txt;

	perceptron(int n_i, int h1, int h2, int h3,int n_o) {
		entrada = n_i;

		hidden1 = h1;
		hidden2 = h2;
		hidden3 = h3;

		salida = n_o;
		
	}

	bool end() {
		bool st = true;
		for (int i = 0; i < n_output.size(); i++) {
			if (n_output[i]->y != n_output[i]->value) {
				st = false;
			}

		}
		return st;
	}

	void ini() {
		int size = 2000;
		int size2 = 100;


		build_dataset(size);
		crear_entrada(0.5);

		crear_hiddenlayer(0.5, hidden1, H1);
		link(H1, n_input, hidden1, entrada);



		crear_hiddenlayer(0.5, hidden2, H2);
		link(H2, H1, hidden2, hidden1);


		crear_hiddenlayer(0.5, hidden3, H3);
		link(H3, H2, hidden3, hidden2);

		
		crear_salida(0.5);
		link(n_output, H3, salida, hidden3);
		build_testingset(size, size2);

	}

	
	void intervals(int interval,int max) {
	
		int t = 0;
		while (t < max) {
			t = t + interval;
			vals.push_back(t);
		}
	}


	bool is_in(int u) {
		for (auto a : vals) {
			if (a == u) {
				return true;

			}
		}
		return false;
	}

	void to_txt(const std::string& filename) {
		std::ofstream outputFile(filename); // Open file for writing

		if (outputFile.is_open()) {
			for (const auto& element : txt) {
				outputFile << element << "\n"; // Write each element followed by a newline
			}
			outputFile.close(); // Close the file
			std::cout << "Elements written to " << filename << " successfully!" << std::endl;
		}
		else {
			std::cerr << "Unable to open file: " << filename << std::endl;
		}
	}

	void train(int max_epochs, double initial_learning_rate) {
		//prep
		double lr = initial_learning_rate;
		int epoch = 0;

		int inter = dataset.size()/ testingset.size();
		int t = 0;

		intervals(inter, dataset.size());
		
		for (int u = 0; u < dataset.size(); u++) {
			
			if (u >=dataset.size()-testingset.size()-20 && t<testingset.size()) {
				test(t);
				t++;
			}
			
			

			epoch = 0;
			while (epoch < max_epochs) {

				for (int i = 0; i < n_input.size(); i++) {
					n_input[i]->value = dataset[u][i + 1];
				}




				n_output[0]->expected = dataset[u][0];

				//TRAIN HERE


				//calculate ys of the layers

				for (int i = 0; i < H1.size(); i++) {
					H1[i]->calculate_y();
				}

				for (int i = 0; i < H2.size(); i++) {
					H2[i]->calculate_y();
				}

				for (int i = 0; i < H3.size(); i++) {
					H3[i]->calculate_y();
				}

				for (int i = 0; i < n_output.size(); i++) {
					n_output[i]->calculate_y2();
				}

				//output neuron error and new weights

				double sum = 0.0;

				for (int i = 0; i < n_output[0]->neuronas.size(); i++) {
					sum += n_output[0]->weights[i] * n_output[0]->neuronas[i]->value;
				}
				double error1 = (n_output[0]->expected - n_output[0]->value) * n_output[0]->act1_derivative(sum);

				n_output[0]->error = error1;
				n_output[0]->new_weights(lr, error1);

				//errores de la H3

				n_output[0]->new_errors(error1);




				//errores para la capa2
				



				//------------
				double sum_err = 0.0;

				for (int j = 0; j < H2.size(); j++) {
					sum = 0.0;

					for (int i = 0; i < H2[j]->weights.size(); i++) {
						sum += H2[j]->weights[i] * H2[j]->neuronas[i]->value;
					}

					sum_err = H3[0]->weights[j] * H3[0]->error;


					H2[j]->error = sum_err * H3[0]->act1_derivative(sum);
					
				}
				H3[0]->weights_hidden(lr);

				//-------------

					sum_err = 0.0;

				

					for (int j = 0; j < H1.size(); j++) {
						sum = 0.0;

						for (int i = 0; i < H1[j]->weights.size(); i++) {
							sum += H1[j]->weights[i] * H1[j]->neuronas[i]->value;
						}

						for (int i = 0; i < H2.size(); i++) {
							sum_err += H2[i]->weights[j] * H2[i]->error;
						}
						H1[j]->error = sum_err * H1[0]->act1_derivative(sum);

					}

					for (int i = 0; i < H2.size(); i++) {
						H2[i]->weights_hidden(lr);
					}
				

				//------------
					sum_err = 0.0;

					for (int i = 0; i < n_input.size(); i++) {
						sum = n_input[i]->value;

						for (int j = 0; j < H1.size(); j++) {
							sum_err += H1[j]->weights[i] * H1[j]->error;
						}
						n_input[i]->error = sum_err * n_input[0]->act1_derivative(sum);
					}

					for (int i = 0; i < H1.size(); i++) {
						H1[i]->weights_hidden(lr);
					}

				//------------

				epoch++;

				///////
				
			}
			int k;
			k = 4;
			
		}
		
		
		to_txt("results.txt");
	}

	void test(int u) {

		

				for (int i = 0; i < n_input.size(); i++) {
					n_input[i]->value = testingset[u][i + 1];
				}
				
				for (int i = 0; i < H1.size(); i++) {
					H1[i]->calculate_y();
				}

				for (int i = 0; i < H2.size(); i++) {
					H2[i]->calculate_y();
				}

				for (int i = 0; i < H3.size(); i++) {
					H3[i]->calculate_y();
				}

				for (int i = 0; i < n_output.size(); i++) {
					n_output[i]->calculate_y3();
					//cout << n_output[i]->value << endl;
					txt.push_back(n_output[i]->value);
				}

				
		

		
	}


	//DATASET---------------

	void build_testingset(int s1, int s2){
		int t = s1 + s2;
		int n = 1000;

		
		
		for (; s1 < t; s1++) {
			testingset.push_back(data(s1));
		}
		
	}




	void build_dataset(int n) {
		
//		n = n + 500;

		
		
		for (int i = 1; i < n; i++) {
			dataset.push_back(data(i));
		}
		
	}


	std::vector<double> data(int i) {
		vector<double> x;
		std::vector<std::string> rowData;


		std::ifstream file("Data.csv");
		if (!file.is_open()) {
			std::cerr << "Error opening the file." << std::endl;
	
		}

		std::string line;
		std::getline(file, line); 

		for (int row = 0; row < i; ++row) {
			if (!std::getline(file, line)) {
				std::cerr << "Error: Row " << i << " not found." << std::endl;
				
			}
		}

		std::istringstream iss(line);
		std::string token;
		while (std::getline(iss, token, ',')) {
			rowData.push_back(token);
		}

		file.close();
		x = tr(rowData);

		return x;
	}



	vector<double> tr(vector<string>& a) {
		vector<double> out;

		a[1].erase(a[1].find('$'), 1);
		out.push_back(stod(a[1]));

		out.push_back(stod(a[2]));

		a[3].erase(a[3].find('$'), 1);
		out.push_back(stod(a[3]));


		a[4].erase(a[4].find('$'), 1);
		out.push_back(stod(a[4]));

		a[5].erase(a[5].find('$'), 1);
		out.push_back(stod(a[5]));

		return out;
	}


	//---------------------


	//INI------------------------
	void crear_entrada(int lr) {
		entry = dataset[0];
		for (int i = 1; i < entrada+1; i++) {
			neuron* a = new neuron('i', entry[i], lr);
			n_input.push_back(a);
		}

	}

	void crear_hiddenlayer(int lr, int hidden, vector<neuron*>& H) {
		for (int i = 0; i < hidden; i++) {
			neuron* a = new neuron('h', 0.0, lr);
			H.push_back(a);
		}

	}


	void crear_salida(int lr) {
		exit = dataset[0];

		for (int i = 0; i < salida; i++) {
			neuron* a = new neuron('o', exit[i], lr);
			n_output.push_back(a);

		}
		n_output.at(0)->expected = exit[0];

	}

	void link(vector<neuron*> v1, vector<neuron*> v2, int s1, int s2) {

		for (int i = 0; i < s1; i++) {
			for (int j = 0; j < s2; j++) {
				v1[i]->add(v2[j]);
				double random_weight;
				random_weight = 0.0;
				//random_weight = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
				v1[i]->weights.push_back(random_weight);
				

			}
		}
	}

	//------------------------


};


int main()
{
	srand(static_cast<unsigned int>(time(NULL)));
	perceptron MLP(4, 20, 15, 1, 1);
	MLP.ini();
	MLP.train(100,0.95);
	//MLP.test();

}

