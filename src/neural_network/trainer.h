#ifndef SRC_NEURAL_NETWORK_TRAINER_H_
#define SRC_NEURAL_NETWORK_TRAINER_H_

#include <cstdint>
#include <utility>

#include "src/params.h"
#include "src/io/csv_reader.h"
#include "src/neural_network/neural_network.h"

NeuralNetwork Train(Parameters params, CsvReader reader);

#endif
