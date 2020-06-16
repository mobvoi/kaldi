// pybind/feat/feature_window_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/feature_window_pybind.h"

#include "feat/feature-window.h"

using namespace kaldi;

static void pybind_frame_extraction_options(py::module& m) {
  using PyClass = FrameExtractionOptions;
  py::class_<FrameExtractionOptions> pyclass(m, "FrameExtractionOptions");
  DEF_INIT();
  DEF(WindowShift);
  DEF(WindowSize);
  DEF(PaddedWindowSize);
  DEF_P(samp_freq);
  DEF_P(frame_shift_ms);
  DEF_P(frame_length_ms);
  DEF_P(dither);
  DEF_P(preemph_coeff);
  DEF_P(remove_dc_offset);
  DEF_P(window_type);
  DEF_P(round_to_power_of_two);
  DEF_P(blackman_coeff);
  DEF_P(snip_edges);
  DEF_P(allow_downsample);
  DEF_P(allow_upsample);
  DEF_P(max_feature_vectors);
  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;
    os << "samp_freq: " << self.samp_freq << "\n";
    os << "frame_shift_ms: " << self.frame_shift_ms << "\n";
    os << "frame_length_ms: " << self.frame_length_ms << "\n";
    os << "dither: " << self.dither << "\n";
    os << "preemph_coeff: " << self.preemph_coeff << "\n";
    os << "remove_dc_offset: " << self.remove_dc_offset << "\n";
    os << "window_type: " << self.window_type << "\n";
    os << "round_to_power_of_two: " << self.round_to_power_of_two << "\n";
    os << "blackman_coeff: " << self.blackman_coeff << "\n";
    os << "snip_edges: " << self.snip_edges << "\n";
    os << "allow_downsample: " << self.allow_downsample << "\n";
    os << "allow_upsample: " << self.allow_upsample << "\n";
    os << "max_feature_vectors: " << self.max_feature_vectors << "\n";
    return os.str();
  });
}

void pybind_feature_window(py::module& m) {
  pybind_frame_extraction_options(m);
}
