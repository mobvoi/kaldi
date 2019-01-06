// nnet3/nnet-chaina-training-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3a/nnet-chaina-training.h"

namespace kaldi {
namespace nnet3 {


void UnitTestCompile() {
  // just testing the compilation works, i.e. that all member functions are
  // defined
  NnetChainaTrainingOptions config;
  NnetChainaModels models(true, false, false, "a", "b", "c");
  NnetChainaTrainer  trainer(config, &models);
}


} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(2);
  // KALDI_LOG << "Tests succeeded.";
  return 0;
}
