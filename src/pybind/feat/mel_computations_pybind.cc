// pybind/feat/mel_computations_pybind.cc

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

#include "feat/mel_computations_pybind.h"

#include "feat/mel-computations.h"

using namespace kaldi;

static void pybind_mel_banks_options(py::module& m) {
  using PyClass = MelBanksOptions;
  DEF_CLASS(MelBanksOptions);
  pyclass.def(py::init<int>(), py::arg("num_bins") = 25);
  DEF_P(num_bins);
  DEF_P(low_freq);
  DEF_P(high_freq);
  DEF_P(vtln_low);
  DEF_P(vtln_high);
  DEF_P(debug_mel);
  DEF_P(htk_mode);
  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;
    os << "num_bins: " << self.num_bins << "\n";
    os << "low_freq: " << self.low_freq << "\n";
    os << "high_freq: " << self.high_freq << "\n";
    os << "vtln_low: " << self.vtln_low << "\n";
    os << "vtln_high: " << self.vtln_high << "\n";
    os << "debug_mel: " << self.debug_mel << "\n";
    os << "htk_mode: " << self.htk_mode << "\n";
    return os.str();
  });
}

void pybind_mel_computations(py::module& m) { pybind_mel_banks_options(m); }
