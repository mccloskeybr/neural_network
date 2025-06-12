#include "src/io/model_checkpoint.h"

#include <fstream>
#include <iostream>
#include <string>

#include "src/common/assert.h"
#include "src/protos/model_checkpoint.pb.h"

protos::ModelCheckpoint ReadModelCheckpoint(std::string file_path) {
  std::fstream stream(file_path, std::ios::in | std::ios::binary);
  protos::ModelCheckpoint checkpoint_proto;
  bool status = checkpoint_proto.ParseFromIstream(&stream);
  ASSERT(status);
  stream.close();
  return checkpoint_proto;
}

void WriteModelCheckpoint(
    std::string file_path, const protos::ModelCheckpoint& checkpoint_proto) {
  std::fstream stream(file_path, std::ios::out | std::ios::binary);
  bool status = checkpoint_proto.SerializeToOstream(&stream);
  ASSERT(status);
}
