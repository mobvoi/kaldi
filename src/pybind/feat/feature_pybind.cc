// pybind/feat/feature_pybind.cc

// Copyright 2019   Microsoft Corporation (author: Xingyu Na)

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

#include "feat/feature_pybind.h"

#include "feat/feature-fbank.h"
#include "feat/feature-mfcc.h"

using namespace kaldi;

template <class Feature>
void offline_feature(py::module& m, const std::string& feat_type) {
  py::class_<OfflineFeatureTpl<Feature>>(m, feat_type.c_str())
      .def(py::init<const typename Feature::Options&>())
      .def("ComputeFeatures", &OfflineFeatureTpl<Feature>::ComputeFeatures)
      .def("Dim", &OfflineFeatureTpl<Feature>::Dim);
}

void pybind_feature(py::module& m) {
  py::class_<MfccOptions>(m, "MfccOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &MfccOptions::frame_opts)
      .def_readwrite("mel_opts", &MfccOptions::mel_opts)
      .def_readwrite("num_ceps", &MfccOptions::num_ceps)
      .def_readwrite("use_energy", &MfccOptions::use_energy)
      .def_readwrite("energy_floor", &MfccOptions::energy_floor)
      .def_readwrite("raw_energy", &MfccOptions::raw_energy)
      .def_readwrite("cepstral_lifter", &MfccOptions::cepstral_lifter)
      .def_readwrite("htk_compat", &MfccOptions::htk_compat);

  offline_feature<MfccComputer>(m, "Mfcc");
  offline_feature<FbankComputer>(m, "Fbank");
}
