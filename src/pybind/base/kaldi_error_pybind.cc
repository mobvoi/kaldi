// pybind/base/kaldi_error_pybind.cc

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

#include "base/kaldi_error_pybind.h"

#include "base/kaldi-error.h"

using namespace kaldi;

static void DummyLogHandler(const LogMessageEnvelope& /*envelope*/,
                            const char* /*message*/) {}

void pybind_kaldi_error(py::module& m) {
  m.def("DisableLog", []() { SetLogHandler(&DummyLogHandler); });
  m.def("EnableLog", []() { SetLogHandler(nullptr); });
}
