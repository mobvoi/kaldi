// pybind/nnet3/nnet_example_utils_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../../COPYING for clarification regarding multiple authors
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

#include "nnet3/nnet_example_utils_pybind.h"

#include "nnet3/nnet-example-utils.h"

using namespace kaldi;
using namespace nnet3;

static void pybind_example_generation_config(py::module& m) {
  using PyClass = ExampleGenerationConfig;
  DEF_CLASS("ExampleGenerationConfig");
  DEF_INIT();

  DEF_P(left_context);
  DEF_P(right_context);
  DEF_P(left_context_initial);
  DEF_P(right_context_final);
  DEF_P(num_frames_overlap);
  DEF_P(frame_subsampling_factor);
  DEF_P(num_frames_str);

  DEF_P(num_frames);
  DEF(ComputeDerived);

  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;

    os << "left_context: " << self.left_context << "\n";
    os << "right_context: " << self.right_context << "\n";
    os << "left_context_initial: " << self.left_context_initial << "\n";
    os << "right_context_final: " << self.right_context_final << "\n";
    os << "num_frames_overlap: " << self.num_frames_overlap << "\n";
    os << "frame_subsampling_factor: " << self.frame_subsampling_factor << "\n";
    os << "num_frames_str: " << self.num_frames_str << "\n";

    return os.str();
  });
}

static void pybind_chunk_time_info(py::module& m) {
  using PyClass = ChunkTimeInfo;
  DEF_CLASS("ChunkTimeInfo");
  DEF_INIT();

  DEF_P(first_frame);
  DEF_P(num_frames);
  DEF_P(left_context);
  DEF_P(right_context);
  DEF_P(output_weights);

  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;
    os << "first_frame: " << self.first_frame << "\n";
    os << "num_frames: " << self.num_frames << "\n";
    os << "left_context: " << self.left_context << "\n";
    os << "right_context: " << self.right_context << "\n";
    return os.str();
  });
}

static void pybind_utterance_splitter(py::module& m) {
  using PyClass = UtteranceSplitter;
  DEF_CLASS("UtteranceSplitter");
  pyclass.def(py::init<const ExampleGenerationConfig&>(), py::arg("config"));
  DEF_REF(Config);
  pyclass.def(
      "GetChunksForUtterance",
      [](PyClass* self, int32 utterance_length) -> std::vector<ChunkTimeInfo> {
        std::vector<ChunkTimeInfo> chunk_info;
        self->GetChunksForUtterance(utterance_length, &chunk_info);
        return chunk_info;
      });

  pyclass.def("LengthsMatch", &PyClass::LengthsMatch, py::arg("utt"),
              py::arg("utterance_length"), py::arg("supervision_length"),
              py::arg("length_tolerance") = 0);

  DEF(ExitStatus);
}

void pybind_nnet_example_utils(py::module& m) {
  pybind_example_generation_config(m);
  pybind_chunk_time_info(m);
  pybind_utterance_splitter(m);

  m.def("srand", [](unsigned int seed) { srand(seed); });
}
