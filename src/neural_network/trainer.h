#ifndef SRC_NEURAL_NETWORK_TRAINER_H_
#define SRC_NEURAL_NETWORK_TRAINER_H_

#include <cstdint>
#include <utility>

#include "src/io/csv_reader.h"
#include "src/neural_network/params.h"
#include "src/neural_network/neural_network.h"

absl::Status Train(
    struct NeuralNetwork& neural_network, const TrainParameters& params,
    std::string train_data_file_path, std::string score_data_file_path,
    std::string out_model_checkpoint_file_path);

#endif
