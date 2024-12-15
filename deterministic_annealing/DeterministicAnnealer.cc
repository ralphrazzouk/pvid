#include <math.h>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <regex>


// Add constants
const double TEMP_SCALE_FACTOR = 0.01; // Adjust this value as needed
const double MIN_TEMP = 1.0;  
const double MAX_TEMP = 4.0;
const double CONVERGENCE_EPSILON = 1e-9;
const int BATCH_SIZE = 1000;

using namespace std;
namespace fs = std::filesystem;

// Helper function to get file number
int getFileNumber(const std::string& filename) {
    std::regex pattern("_([0-9]+)\\.json$");
    std::smatch matches;
    if (std::regex_search(filename, matches, pattern) && matches.size() > 1) {
        return std::stoi(matches[1]);
    }
    return 0;
}

pair<vector<double>, vector<double>> loadTracks(string filename) {
    ifstream trackFile(filename.c_str());
    vector<double> X;
    vector<double> errorX;
    
    if (!trackFile.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return make_pair(X, errorX);
    }

    string line, full_content;
    while (getline(trackFile, line)) {
        // Remove whitespace
        line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
        full_content += line;
    }
    cout << "Cleaned content: " << full_content.substr(0, 100) << endl;

    size_t pos = 0;
    while ((pos = full_content.find("[[", pos)) != string::npos) {
        pos += 2;  // Skip [[
        // Skip to track array
        pos = full_content.find("[", pos);
        if (pos == string::npos) break;
        
        // Process track pairs
        while (pos < full_content.length()) {
            pos = full_content.find("[", pos + 1);
            if (pos == string::npos) break;
            
            size_t end = full_content.find("]", pos);
            if (end == string::npos) break;
            
            string track = full_content.substr(pos + 1, end - pos - 1);
            size_t comma = track.find(",");
            
            if (comma != string::npos) {
                try {
                    double pos_val = stod(track.substr(0, comma));
                    double err_val = stod(track.substr(comma + 1));
                    X.push_back(pos_val);
                    errorX.push_back(err_val);
                    cout << "Track: " << pos_val << ", " << err_val << endl;
                } catch (const exception& e) {
                    continue;
                }
            }
            
            if (full_content.find("]]", end) < full_content.find("[", end)) {
                pos = full_content.find("[[", end);
                if (pos == string::npos) break;
            }
        }
    }

    cout << "Total tracks: " << X.size() << endl;
    return make_pair(X, errorX);
}

double getDistortion(const double& x, const double& errorX, const double& y) {
	double sigma = (x - y) / errorX;
	return sigma * sigma;
}

double getClusterProbability(int& j, const int& N, vector<vector<double>>& associationMatrix) {
	double probability = 0.0;
	for (int i = 0; i < N; ++i) {
		probability += associationMatrix[j][i];
	}
	probability /= N;
	return probability;
}

std::vector<double> getCriticalTemperatures(vector<double>& X, vector<double>& errorX, vector<double>& Y, vector<vector<double>>& associationMatrix, const int& N) {
	vector<double> criticalTemperatures = {};
	for (int j = 0; j < Y.size(); ++j) {
		double prob_y = getClusterProbability(j, N, associationMatrix);

		double criticalTemperature = 0;
		for (int i = 0; i < N; ++i) {
			criticalTemperature += associationMatrix[j][i] * getDistortion(X[i], errorX[i], Y[j]) / prob_y;
		}
		criticalTemperature /= N;
		criticalTemperatures.push_back(criticalTemperature);
	}

	return criticalTemperatures;
}

bool merge(vector<double>& Y, vector<double>& X, vector<double>& errorX, vector<double>& clusterProbabilities, vector<vector<double>>& associationMatrix, double& T, const int& N) {
	for (int j = 0; (j + 1) < Y.size(); ++j) {
		for (int k = j + 1; k < Y.size(); ++k) {
			if (fabs(Y[j] - Y[k]) < 0.005) { // Need to merge
				// First check to make sure that the merged critical temperature isn't higher than current temperature
				double totalProbability = clusterProbabilities[j] + clusterProbabilities[k];

				double newCentroid = 0.0;
				if (totalProbability > 0) {
					newCentroid = (clusterProbabilities[j] * Y[j] + clusterProbabilities[k] * Y[k]) / totalProbability;
				}
				else {
					newCentroid = 0.5 * (Y[j] + Y[k]);
				}

				double newCriticalTemp = 0.0;
				for (int i = 0; i < N; ++i) {
					newCriticalTemp += (associationMatrix[j][i] + associationMatrix[k][i]) * getDistortion(X[i], errorX[i], newCentroid) / totalProbability;
				}
				newCriticalTemp /= N;

				//printf("Attempting to merge clusters %i and %i, new critical temperature is %f. \n", j, k, newCriticalTemp);

				// If merged cluster critical temperature is greater than the current temperature, skip this merge
				if (newCriticalTemp > T) {
					continue;
				}

				clusterProbabilities[j] = totalProbability;
				Y[j] = newCentroid;
				for (int i = 0; i < N; ++i) {
					associationMatrix[j][i] += associationMatrix[k][i];
				}
				//printf("Merged clusters %i and %i \n", j, k);
				// Now delete the merged cluster
				Y.erase(Y.begin() + k);
				clusterProbabilities.erase(clusterProbabilities.begin() + k);
				associationMatrix.erase(associationMatrix.begin() + k);
				return true;
			}
		}
	}
	return false;
}

double getPartitionFunction(double& x, double& errorx, vector<double>& Y, double& T, vector<double>& clusterProbabilities, vector<double>& partitionComponents) {
    double partition_func = 0.0;
    for (int i = 0; i < Y.size(); ++i) {
        // Add protection against overflow
        double distortion = getDistortion(x, errorx, Y[i]);
        double exp_term = clusterProbabilities[i] * exp(-1 * distortion / T);
        
        // Protect against overflow
        if (exp_term > DBL_MAX/2) {
            exp_term = DBL_MAX/2;
        }
        
        partitionComponents.push_back(exp_term);
        partition_func += exp_term;
    }
    return partition_func;
}

void updateAssociationProbabilityMatrix(vector<vector<double>>& associationMatrix, vector<double>& X, vector<double>& errorX, vector<double>& Y, double& T, const int& N, vector<double>& clusterProbabilities, vector<double>& partitionComponents) {
	for (int i = 0; i < N; ++i) {
		double partitionFunc = getPartitionFunction(X[i], errorX[i], Y, T, clusterProbabilities, partitionComponents);
		for (int j = 0; j < Y.size(); ++j) {
			associationMatrix[j][i] = partitionComponents[j] / partitionFunc;
		}
		partitionComponents.clear();
	}
}

double getCentroid(vector<double>& X, vector<double>& jthClusterAssociationProbs, double& jthClusterProbability, const int& N) {
	double centroid = 0.0;
	for (int i = 0; i < N; ++i) {
		centroid += X[i] * jthClusterAssociationProbs[i];
	}
	centroid /= (N * jthClusterProbability);

	return centroid;
}

/**
 * Updates the association matrix, cluster centroids, and cluster probabilities while returning the squared difference in the cluster centroids
 * @param clusterCentroids The vector containing all of the cluster centroids
 * @param associationMatrix The matrix telling what track is associated with what cluster
 * @param clusterProbabilities The vector containing all of the probabilities that we would split into each cluster
 * @param X The data that is being clustered
 * @param errorX The error on the data being clustered
 * @param T The current temperature
 *
 * @return The vector containing the squared difference in cluster centroids between old and new centroids
 */
vector<double> update(vector<double>& clusterCentroids, vector<vector<double>>& associationMatrix, 
                     vector<double>& clusterProbabilities, vector<double>& X, vector<double>& errorX, 
                     double& T, const int& N, vector<double>& partitionComponents) {
    
    vector<double> deltas;
    
    // Process in batches
    for(int start = 0; start < N; start += BATCH_SIZE) {
        int end = min(start + BATCH_SIZE, N);
        
        // Update association probabilities for this batch
        for(int i = start; i < end; i++) {
            double partitionFunc = getPartitionFunction(X[i], errorX[i], clusterCentroids, T, 
                                                      clusterProbabilities, partitionComponents);
            
            // Protect against division by zero
            if(partitionFunc < DBL_MIN) {
                partitionFunc = DBL_MIN;
            }
            
            for(int j = 0; j < clusterCentroids.size(); j++) {
                associationMatrix[j][i] = partitionComponents[j] / partitionFunc;
            }
            partitionComponents.clear();
        }
    }

    // Update cluster centroids
    vector<double> oldCentroids = clusterCentroids;
    for(int j = 0; j < clusterProbabilities.size(); j++) {
        clusterProbabilities[j] = getClusterProbability(j, N, associationMatrix);
        
        // Protect against zero probability
        if(clusterProbabilities[j] < DBL_MIN) {
            clusterProbabilities[j] = DBL_MIN;
        }
        
        double newCentroid = getCentroid(X, associationMatrix[j], clusterProbabilities[j], N);
        double deltaCentroid = clusterCentroids[j] - newCentroid;
        deltas.push_back(deltaCentroid * deltaCentroid);
        clusterCentroids[j] = newCentroid;
    }
    
    return deltas;
}

bool split(vector<double>& X, vector<double>& errorX, vector<double>& clusterCentroids, vector<vector<double>>& associationMatrix, vector<double>& clusterProbabilities, double& T, const int& N, double delta = 1e-3) {
	vector<double> criticalTemps = getCriticalTemperatures(X, errorX, clusterCentroids, associationMatrix, N);
	//stable_sort(criticalTemps.begin(), criticalTemps.end(), std::greater<double>() );
	/*printf("The current cluster critical temperatures are: \n");
	for (int i = 0; i < criticalTemps.size(); ++i) {
		printf("\t Temperature %i: %f \n", i, criticalTemps[i]);
	}
	printf("\n");*/

	bool split = false;

	for (int k = 0; k < criticalTemps.size(); ++k) {
		if (T <= criticalTemps[k]) { // need to split that cluster
			split = true;
			//printf("Splitting the %ith cluster. \n", k);
			double leftClusterProb = 0.0; // new cluster formed from tracks whose z < old centroid
			double rightClusterProb = 0.0; // new cluster formed from tracks whose z > old centroid

			double leftTotalWeight = 0.0;
			double rightTotalWeight = 0.0;
			double leftCentroid = 0.0;
			double rightCentroid = 0.0;

			for (int i = 0; i < N; ++i) {
				double probabilty = associationMatrix[k][i];
				double errorx = errorX[i];
				double x = X[i];
				double weight = probabilty / (errorx * errorx);

				if (x < clusterCentroids[k]) {
					leftClusterProb += probabilty;
					leftTotalWeight += weight;
					leftCentroid += weight * x;
				}
				else {
					rightClusterProb += probabilty;
					rightTotalWeight += weight;
					rightCentroid += weight * x;
				}
			}

			if (leftTotalWeight > 0) {
				leftCentroid = leftCentroid / leftTotalWeight;
			}
			else {
				leftCentroid = clusterCentroids[k] - delta;
			}

			if (rightTotalWeight > 0) {
				rightCentroid = rightCentroid / rightTotalWeight;
			}
			else {
				rightCentroid = clusterCentroids[k] + delta;
			}

			// TODO: Reduce split size if there is not enough room
			// reduce split size if there is not enough room
			/*if( ( ik   > 0       ) && ( y[ik-1].z>=z1 ) ){ z1=0.5*(y[ik].z+y[ik-1].z); }
			if( ( ik+1 < y.size()) && ( y[ik+1].z<=z2 ) ){ z2=0.5*(y[ik].z+y[ik+1].z); }*/

			if (rightCentroid - leftCentroid > delta) {
				clusterProbabilities.push_back(leftClusterProb * clusterProbabilities[k] / (leftClusterProb + rightClusterProb));

				vector<double> newAssociationProbs = {};
				for (int i = 0; i < N; ++i) {
					newAssociationProbs.push_back(leftClusterProb * associationMatrix[k][i] / (leftClusterProb + rightClusterProb));
				}
				associationMatrix.push_back(newAssociationProbs);
				clusterCentroids.push_back(leftCentroid);

				clusterProbabilities[k] = rightClusterProb * clusterProbabilities[k] / (leftClusterProb + rightClusterProb);
				clusterCentroids[k] = rightCentroid;
				for (int i = 0; i < N; ++i) {
					associationMatrix[k][i] = rightClusterProb * associationMatrix[k][i] / (leftClusterProb + rightClusterProb);
				}
			}

			// old splitting code
			/*
			clusterCentroids.push_back(clusterCentroids[k] + delta);
			vector<double> newAssociationProbs = {};
			for (int i = 0; i < N; ++i) {
				newAssociationProbs.push_back(associationMatrix[k][i] / 2.0);
			}
			associationMatrix.push_back(newAssociationProbs);
			clusterProbabilities.push_back(clusterProbabilities[k] / 2.0);
			clusterProbabilities[k] /= 2.0;
			for (int i = 0; i < N; ++i) {
				associationMatrix[k][i] /= 2.0;
			}*/
		}
	}
	return split;
}

double S(const vector<double>& centroids, const vector<double>& X, 
         const vector<double>& errorX, const vector<double>& clusterProbabilities,
         const double T, const int N) {
    double energy = 0.0;
    
    for(int i = 0; i < N; i++) {
        double partial_sum = 0.0;
        
        for(int k = 0; k < centroids.size(); k++) {
            double distortion = getDistortion(X[i], errorX[i], centroids[k]);
            double exp_term = clusterProbabilities[k] * exp(-distortion / T);
            
            if(exp_term > DBL_MAX/2) {
                exp_term = DBL_MAX/2;
            }
            
            partial_sum += exp_term;
        }
        
        if(partial_sum < DBL_MIN) {
            partial_sum = DBL_MIN;
        }
        
        energy -= log(partial_sum);
    }
    
    energy *= T;
    
    return energy;
}

int main() {
	std::string outputDir = "./test/set1/jsons/";
    try {
        fs::create_directories(outputDir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return 1;
    }

    // Get list of input files
    std::string inputDir = "../datasets/test/set1/";
    std::vector<std::string> inputFiles;
    
    try {
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".json" && 
                entry.path().filename() != "serializedEvents.json") {
                inputFiles.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading input directory: " << e.what() << std::endl;
        return 1;
    }

    // Sort files to ensure consistent processing order
    std::sort(inputFiles.begin(), inputFiles.end());

    // Process each file
	for (const auto& inputFile : inputFiles) {
		std::cout << "Processing " << inputFile << std::endl;
        
        // Get data from file
        pair<vector<double>, vector<double>> data = loadTracks(inputFile);
        vector<double> X = data.first;
        vector<double> errorX = data.second;

        if (X.empty()) {
            std::cerr << "No tracks loaded from " << inputFile << std::endl;
            continue;
        }

        std::clock_t time_start = std::clock();

		// Step 1: Set Limts
		// Initial temperature parameters
		double Tmin = 4;  // Increase from original to prevent getting stuck
		double betaMax = 1.0/Tmin;
		double Tstop = 1.0;
		double coolingFactor = 0.8;  // Make cooling more aggressive (was 0.6)
		double nSweeps = 100;  // Reduce from 250 to speed up
		bool useLinearCooling = true;
		double delta = 3.3e-5;
		int maxIterations = 100;  // Reduce from 350
		double convergenceCriteria = 1e-6; // Relax from 1e-9
		int Kmax = 2; //TODO: Set this to nVertices
		int N = X.size();

		// Step 2: Initialize
		vector<double> clusterProbabilities = { 1.0 };
		vector<double> partitionComponents = {};

		vector<double> clusterCentroids = { 0.0 };
		for (int j = 0; j < clusterCentroids.size(); ++j) {
			for (int i = 0; i < N; ++i) {
				clusterCentroids[j] += X[i];
			}
			clusterCentroids[j] /= N;
		}

		vector<vector<double>> associationMatrix = { {} };
		for (int i = 0; i < N; ++i) {
			associationMatrix[0].push_back(1.0);
		}

		// Set initial temperature to first critical temperature
		double T = getCriticalTemperatures(X, errorX, clusterCentroids, associationMatrix, N)[0];
		double beta = 1.0/T;

		// Add temperature scaling based on number of tracks
		T = T * (1.0 + log(N) * TEMP_SCALE_FACTOR);
		if(T < MIN_TEMP) T = MIN_TEMP;
		if(T > MAX_TEMP) T = MAX_TEMP;

		double deltaBeta = (betaMax - beta) / nSweeps;
		//printf("The first critical temperature is at %f \n", T);

		// Add a maximum time limit
		const clock_t start_time = clock();
		const double TIME_LIMIT = 300.0; // 5 minutes in seconds

		// Annealing Loop
		size_t total_iterations = 0;
		while (T > Tmin) {
			// Check time limit
			double elapsed_time = (clock() - start_time) / (double)CLOCKS_PER_SEC;
			if (elapsed_time > TIME_LIMIT) {
				cout << "Time limit reached, terminating annealing." << endl;
				break;
			}

			// Get the state into equilibrium
			for (int n = 0; n < maxIterations; n++) {
				total_iterations++;
				
				vector<double> deltas = update(clusterCentroids, associationMatrix, clusterProbabilities, 
											X, errorX, T, N, partitionComponents);

				// Check for convergence
				double sum = 0.0;
				for (int j = 0; j < deltas.size(); j++) {
					sum += deltas[j];
				}

				if (sum < convergenceCriteria || total_iterations > 10000) {
					break;
				}
			}

			// Check for merging
			while (merge(clusterCentroids, X, errorX, clusterProbabilities, associationMatrix, T, N)) {
				update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);
			}

			// Cool the temperature more aggressively
			if (useLinearCooling) {
				beta += deltaBeta;
				T = 1.0 / beta;
			} else {
				T *= coolingFactor;
			}
			
			// Print progress occasionally
			if (total_iterations % 10 == 0) {
				cout << "Temperature: " << T << ", Iterations: " << total_iterations << endl;
			}
		}

		// Do your final splitting before you anneal down to the final/stopping temperature
		// There is no splitting after this

		update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents); // update first
		// Check for merging
		while (merge(clusterCentroids, X, errorX, clusterProbabilities, associationMatrix, T, N)) {
			update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);
		}
		unsigned int ntry = 0;
		while (split(X, errorX, clusterCentroids, associationMatrix, clusterProbabilities, T, N, delta) && ntry++ < 10) {
			// Get the state into equilibrium
			for (int n = 0; n < maxIterations; ++n) {
				vector<double> deltas = update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);

				// Check for convergence
				double sum = 0.0;
				for (int j = 0; j < deltas.size(); ++j) {
					sum += deltas[j];
				}

				if (sum < convergenceCriteria) {
					break;
				}
			}

			merge(clusterCentroids, X, errorX, clusterProbabilities, associationMatrix, T, N);
			update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents); // update first
		}

		T = Tstop;
		// Step 5
		// Get the state into equilibrium for the final temperature
		for (int n = 0; n < maxIterations; ++n) {
			vector<double> deltas = update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);

			// Check for convergence
			double sum = 0.0;
			for (int j = 0; j < deltas.size(); ++j) {
				sum += deltas[j];
			}

			if (sum < convergenceCriteria) {
				break;
			}
		}

		// Check for merging at the end
		while (merge(clusterCentroids, X, errorX, clusterProbabilities, associationMatrix, T, N)) {
			vector<double> deltas = update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);
		}

		// do a final update on paramaters since some might have merged
		vector<double> deltas = update(clusterCentroids, associationMatrix, clusterProbabilities, X, errorX, T, N, partitionComponents);

		std::clock_t time_stop = std::clock();

		printf("The current centroid locations are: \n");
		for (int i = 0; i < clusterCentroids.size(); ++i) {
			printf("\t Centroid %i: %f \n", i, clusterCentroids[i]);
		}
		printf("\n");

		printf("The current centroid probabilities are: \n");
		for (int i = 0; i < clusterProbabilities.size(); ++i) {
			printf("\t Probability %i: %f \n", i, clusterProbabilities[i]);
		}
		printf("\n");

		int fileNum = getFileNumber(inputFile);
        std::string outputPath = outputDir + "serializableResponse_" + std::to_string(fileNum) + ".json";
        std::ofstream responseFile(outputPath);

        if (!responseFile.is_open()) {
            std::cerr << "Failed to open output file: " << outputPath << std::endl;
            continue;
        }

        double energy = 0.0;
        responseFile << "[";
        responseFile << "[" << energy << ", [";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < clusterProbabilities.size(); ++j) {
                responseFile << "\"" << ((int)round(associationMatrix[j][i])) << "\"";
                if (i == N - 1 && j == clusterProbabilities.size() - 1) {
                    responseFile << "]";
                } else {
                    responseFile << ", ";
                }
            }
        }
        responseFile << "], " << (time_stop - time_start) << "]";
        responseFile.close();

        std::cout << "Completed processing " << inputFile << " -> " << outputPath << std::endl;
	}

	std::cout << "All files processed successfully" << std::endl;
    return 0;

}
