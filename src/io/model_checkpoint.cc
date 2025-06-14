#include "src/io/model_checkpoint.h"

#include <fstream>
#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/protos/model_checkpoint.pb.h"

absl::StatusOr<protos::ModelCheckpoint> ReadModelCheckpoint(std::string file_path) {
  std::fstream stream(file_path, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Error opening file with path: ", file_path));
  }
  protos::ModelCheckpoint checkpoint_proto;
  bool status = checkpoint_proto.ParseFromIstream(&stream);
  if (!status) {
    return absl::InvalidArgumentError(
        absl::StrCat("Error reading file checkpoint from file: ", file_path));
  }
  stream.close();
  return checkpoint_proto;
}

absl::Status WriteModelCheckpoint(
    std::string file_path, const protos::ModelCheckpoint& checkpoint_proto) {
  std::fstream stream(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!stream.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Error opening file with path: ", file_path));
  }
  bool status = checkpoint_proto.SerializeToOstream(&stream);
  if (!status) {
    return absl::InvalidArgumentError(
        absl::StrCat("Error writing file checkpoint to file: ", file_path));
  }
  stream.close();
  return absl::OkStatus();
}
