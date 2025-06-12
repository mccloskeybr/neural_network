#ifndef SRC_IO_MODEL_CHECKPOINT_H_
#define SRC_IO_MODEL_CHECKPOINT_H_

#include <string>

#include "src/protos/model_checkpoint.pb.h"

protos::ModelCheckpoint ReadModelCheckpoint(std::string file_path);
void WriteModelCheckpoint(protos::ModelCheckpoint checkpoint_proto);
void WriteModelCheckpoint(
    std::string file_path, const protos::ModelCheckpoint& checkpoint_proto);

#endif
