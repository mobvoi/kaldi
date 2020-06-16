// pybind/feat/feature_common_pybind.h

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

#ifndef KALDI_PYBIND_FEAT_FEATURE_COMMON_PYBIND_H_
#define KALDI_PYBIND_FEAT_FEATURE_COMMON_PYBIND_H_

#include "pybind/kaldi_pybind.h"

template <typename F>
void pybind_offline_feature(py::module& m, const char* name) {
  using namespace kaldi;
  using PyClass = OfflineFeatureTpl<F>;
  using Options = typename PyClass::Options;
  DEF_CLASS(name);
  pyclass.def(py::init<const Options&>(), py::arg("opts"));
  DEF(Dim);
  pyclass.def("ComputeFeatures",
              [](PyClass* self, const VectorBase<BaseFloat>& wave,
                 BaseFloat sample_freq, BaseFloat vtln_warp) {
                Matrix<BaseFloat> features;
                self->ComputeFeatures(wave, sample_freq, vtln_warp, &features);
                return features;
              });
}

#endif  // KALDI_PYBIND_FEAT_FEATURE_COMMON_PYBIND_H_
