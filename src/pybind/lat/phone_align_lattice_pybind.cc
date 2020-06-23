// pybind/lat/phone_align_lattice_pybind.cc

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

#include "lat/phone_align_lattice_pybind.h"

#include "lat/lattice-functions.h"
#include "lat/phone-align-lattice.h"

using namespace kaldi;

static void pybind_phone_align_lattice_options(py::module& m) {
  using PyClass = PhoneAlignLatticeOptions;
  DEF_CLASS("PhoneAlignLatticeOptions");

  DEF_INIT();
  DEF_P(reorder);
  DEF_P(remove_epsilon);
  DEF_P(replace_output_symbols);

  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;
    os << "reorder: " << self.reorder << "\n";
    os << "remove_epsilon: " << self.remove_epsilon << "\n";
    os << "replace_output_symbols: " << self.replace_output_symbols << "\n";
    return os.str();
  });
}

void pybind_phone_align_lattice(py::module& m) {
  pybind_phone_align_lattice_options(m);
  m.def(
      "PhoneAlignLattice",
      [](const CompactLattice& lat, const TransitionModel& tmodel,
         const PhoneAlignLatticeOptions& opts)
          -> std::pair<bool, CompactLattice> {
        bool is_ok = false;
        CompactLattice lat_out;
        is_ok = PhoneAlignLattice(lat, tmodel, opts, &lat_out);
        if (is_ok && lat_out.Start() != fst::kNoStateId) {
          TopSortCompactLatticeIfNeeded(&lat_out);
        }
        return std::make_pair(is_ok, lat_out);
      },
      py::arg("lat"), py::arg("tmodel"), py::arg("opts"));
}
