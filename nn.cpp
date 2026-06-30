#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
using namespace std;


struct Dataset {
    vector<vector<double>> inputs;
    vector<vector<double>> targets;
};

vector<double> parseCSVLine(string line) {
    vector<double> values;
    stringstream ss(line);
    string val;
    while (getline(ss, val, ',')) {
        try {
            values.push_back(stod(val));    // convert string token to double
        } catch (...) {
            values.push_back(0.0);
        }
    }
    return values;
}

Dataset loadCSV(const string& filename, int targetSize) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row = parseCSVLine(line);

        if (row.size() <= (size_t)targetSize) continue;

        vector<double> input;
        vector<double> target;

        size_t inputSize = row.size() - targetSize;
        for (size_t i = 0; i < inputSize; ++i)     input.push_back(row[i]);
        for (size_t i = inputSize; i < row.size(); ++i) target.push_back(row[i]);

        data.inputs.push_back(input);
        data.targets.push_back(target);
    }
    return data;
}

// Fix 4: split dataset chronologically into train and test
//        trainRatio=0.8 means 80% train, 20% test
void trainTestSplit(const Dataset& data, Dataset& train, Dataset& test, double trainRatio = 0.8) {
    size_t n = data.inputs.size();
    size_t trainSize = (size_t)(n * trainRatio);

    for (size_t i = 0; i < n; ++i) {
        if (i < trainSize) {
            train.inputs.push_back(data.inputs[i]);
            train.targets.push_back(data.targets[i]);
        } else {
            test.inputs.push_back(data.inputs[i]);
            test.targets.push_back(data.targets[i]);
        }
    }
    cout << "Split: " << train.inputs.size() << " train, "
         << test.inputs.size()  << " test samples." << endl;
}

// Fix 4: normalization now fits on train set only and returns the
//        min/max so the same scale can be applied to the test set
void normalizeDataset(Dataset& trainData,
                      Dataset& testData,
                      vector<double>& minVal,
                      vector<double>& maxVal) {
    if (trainData.inputs.empty()) return;

    size_t numInputs = trainData.inputs[0].size();
    minVal.assign(numInputs,  1e9);
    maxVal.assign(numInputs, -1e9);

    // Compute min/max from TRAIN only (no leakage from test)
    for (const auto& row : trainData.inputs) {
        for (size_t i = 0; i < numInputs; ++i) {
            if (row[i] < minVal[i]) minVal[i] = row[i];
            if (row[i] > maxVal[i]) maxVal[i] = row[i];
        }
    }

    auto applyNorm = [&](Dataset& d) {
        for (auto& row : d.inputs) {
            for (size_t i = 0; i < numInputs; ++i) {
                double range = maxVal[i] - minVal[i];
                row[i] = (range != 0.0) ? (row[i] - minVal[i]) / range : 0.0;
            }
        }
    };

    applyNorm(trainData);
    applyNorm(testData);   // same scale, fitted only on train
    cout << "Data normalized (fit on train set)." << endl;
}


inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoidDerivative(double z) {   // Fix 3: renamed x -> z (pre-activation sum)
    return z * (1.0 - z);
}

// Xavier uniform init: Uniform(-1/sqrt(fanIn), 1/sqrt(fanIn))
// keeps rand() but scales the range by fan-in instead of flat (-1, 1)
inline double xavierWeight(int fanIn) {
    double limit = 1.0 / sqrt((double)fanIn);
    double r = (double)rand() / RAND_MAX;   // r in [0, 1]
    return r * 2.0 * limit - limit;         // scale to [-limit, +limit]
}

class NeuralNetwork {
private:
    vector<int> topology;

    vector<vector<double>> neurons;   // post-activation values (a)
    vector<vector<double>> preAct;    // Fix 3: stores pre-activation sums (z), used in comments
    vector<vector<double>> biases;

    vector<vector<vector<double>>> weights;   // [layer][curr_neuron][prev_neuron]

    double learningRate;

public:
    NeuralNetwork(const vector<int>& topology, double learningRate = 0.1)
        : topology(topology), learningRate(learningRate)
    {
        srand(time(NULL));   // seed random number generator

        for (size_t i = 0; i < topology.size(); ++i) {
            int numNeurons = topology[i];

            neurons.push_back(vector<double>(numNeurons, 0.0));
            preAct.push_back(vector<double>(numNeurons, 0.0));  // Fix 3
            biases.push_back(vector<double>(numNeurons, 0.0));

            vector<vector<double>> layerWeights;

            if (i > 0) {
                int prevNeurons = topology[i - 1];
                for (int j = 0; j < numNeurons; ++j) {
                    biases[i][j] = 0.0;   // biases start at 0 (common practice)

                    vector<double> neuronWeights;
                    for (int k = 0; k < prevNeurons; ++k) {
                        // Fix 2: Xavier init based on fan-in (prevNeurons)
                        neuronWeights.push_back(xavierWeight(prevNeurons));
                    }
                    layerWeights.push_back(neuronWeights);
                }
            }
            weights.push_back(layerWeights);
        }

        cout << "Network initialized, structure: ";
        for (int n : topology) cout << n << " ";
        cout << endl;
    }

    void feedForward(const vector<double>& input) {
        if (input.size() != (size_t)topology[0]) {
            cerr << "Error: Input size mismatch." << endl;
            return;
        }

        neurons[0] = input;

        for (size_t i = 1; i < topology.size(); ++i) {
            int prevLayerSize = topology[i - 1];

            for (int j = 0; j < topology[i]; ++j) {
                double z = 0.0;                                 // Fix 3: z = pre-activation sum

                for (int k = 0; k < prevLayerSize; ++k) {
                    z += neurons[i-1][k] * weights[i][j][k];
                }
                z += biases[i][j];

                preAct[i][j]  = z;                             // Fix 3: store z
                neurons[i][j] = sigmoid(z);                    // Fix 3: a = sigmoid(z)
            }
        }
    }

    void backPropagate(const vector<double>& target) {
        if (target.size() != (size_t)topology.back()) {
            cerr << "Error: Target size mismatch." << endl;
            return;
        }

        vector<vector<double>> deltas(topology.size());

        // Output layer delta: (target - a) * sigmoid'(a)
        int lastIdx = topology.size() - 1;
        deltas[lastIdx].resize(topology[lastIdx]);

        for (int i = 0; i < topology[lastIdx]; ++i) {
            double a = neurons[lastIdx][i];
            deltas[lastIdx][i] = (target[i] - a) * sigmoidDerivative(a);
        }

        // Hidden layer deltas: sum(delta_next * w_connecting) * sigmoid'(a)
        for (int i = lastIdx - 1; i > 0; --i) {
            deltas[i].resize(topology[i]);
            int nextLayerSize = topology[i + 1];

            for (int j = 0; j < topology[i]; ++j) {
                double errorSum = 0.0;
                for (int k = 0; k < nextLayerSize; ++k) {
                    errorSum += deltas[i+1][k] * weights[i+1][k][j];
                }
                deltas[i][j] = errorSum * sigmoidDerivative(neurons[i][j]);
            }
        }

        // Weight and bias update
        for (size_t i = 1; i < topology.size(); ++i) {
            for (int j = 0; j < topology[i]; ++j) {
                biases[i][j] += learningRate * deltas[i][j];

                for (int k = 0; k < topology[i-1]; ++k) {
                    weights[i][j][k] += learningRate * deltas[i][j] * neurons[i-1][k];
                }
            }
        }
    }

    vector<double> getOutput() {
        return neurons.back();
    }

    // Fix 4: evaluate MSE on a dataset without updating weights
    double evaluate(const Dataset& data) {
        double totalError = 0.0;
        for (size_t i = 0; i < data.inputs.size(); ++i) {
            feedForward(data.inputs[i]);
            vector<double> out = getOutput();
            double err = data.targets[i][0] - out[0];
            totalError += err * err;
        }
        return totalError / data.inputs.size();
    }
};

int main() {
    string csvFile = "data.csv";

    vector<int> topology       = {2, 3, 1};
    int numTargetColumns       = 1;

    Dataset allData = loadCSV(csvFile, numTargetColumns);

    if (allData.inputs.empty()) {
        cerr << "No data loaded." << endl;
        return 1;
    }

    // Fix 4: chronological 80/20 split before normalization
    Dataset trainData, testData;
    trainTestSplit(allData, trainData, testData, 0.8);

    // Fix 4: normalize — fit on train, apply same scale to test
    vector<double> minVal, maxVal;
    normalizeDataset(trainData, testData, minVal, maxVal);

    if (trainData.inputs[0].size() != (size_t)topology[0]) {
        cerr << "Error: CSV input columns (" << trainData.inputs[0].size()
             << ") do not match input layer size (" << topology[0] << ")." << endl;
        return 1;
    }

    int epochs = 50000;
    NeuralNetwork nn(topology, 0.5);

    cout << "\nTraining for " << epochs << " epochs..." << endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < trainData.inputs.size(); ++i) {
            nn.feedForward(trainData.inputs[i]);
            nn.backPropagate(trainData.targets[i]);

            vector<double> out = nn.getOutput();
            double err = trainData.targets[i][0] - out[0];
            totalError += err * err;
        }

        if ((epoch + 1) % 1000 == 0) {
            // Fix 4: report both train and test error every 1000 epochs
            double testError = nn.evaluate(testData);
            cout << "Epoch " << epoch + 1
                 << " | Train MSE: " << totalError / trainData.inputs.size()
                 << " | Test MSE: "  << testError << endl;
        }
    }

    // Fix 4: final predictions shown separately for train and test
    cout << "\n--- Train Set Predictions ---" << endl;
    cout << left << setw(25) << "Input" << setw(10) << "Target" << "Predicted" << endl;
    cout << string(44, '-') << endl;

    for (size_t i = 0; i < trainData.inputs.size(); ++i) {
        nn.feedForward(trainData.inputs[i]);
        vector<double> result = nn.getOutput();

        cout << "[";
        for (double val : trainData.inputs[i])
            cout << fixed << setprecision(4) << val << " ";   // Fix 3: 4dp so values are readable
        cout << "]  "
             << fixed << setprecision(0) << trainData.targets[i][0] << "\t"
             << fixed << setprecision(4) << result[0] << endl;
    }

    cout << "\n--- Test Set Predictions ---" << endl;
    cout << left << setw(25) << "Input" << setw(10) << "Target" << "Predicted" << endl;
    cout << string(44, '-') << endl;

    for (size_t i = 0; i < testData.inputs.size(); ++i) {
        nn.feedForward(testData.inputs[i]);
        vector<double> result = nn.getOutput();

        cout << "[";
        for (double val : testData.inputs[i])
            cout << fixed << setprecision(4) << val << " ";
        cout << "]  "
             << fixed << setprecision(0) << testData.targets[i][0] << "\t"
             << fixed << setprecision(4) << result[0] << endl;
    }

    return 0;
}