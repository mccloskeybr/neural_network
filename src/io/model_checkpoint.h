#ifndef SRC_IO_MODEL_CHECKPOINT_H_
#define SRC_IO_MODEL_CHECKPOINT_H_

#include <string>

#include "src/protos/model_checkpoint.pb.h"

absl::StatusOr<protos::ModelCheckpoint> ReadModelCheckpoint(
    std::string file_path);
absl::Status WriteModelCheckpoint(
    std::string file_path, const protos::ModelCheckpoint& checkpoint_proto);

#endif
