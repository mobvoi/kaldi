// pybind/tree/context_dep_pybind.h

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

#ifndef KALDI_PYBIND_TREE_CONTEXT_DEP_PYBIND_H_
#define KALDI_PYBIND_TREE_CONTEXT_DEP_PYBIND_H_

#include "pybind/kaldi_pybind.h"
#include "tree/context-dep.h"

using namespace kaldi;

void pybind_context_dependency(py::module& m) {
  using PyClass = ContextDependency;
  py::class_<PyClass, ContextDependencyInterface> pyclass(m,
                                                          "ContextDependency");
  DEF_INIT();
  DEF(Read);
  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;
    self.Write(os, false);
    return os.str();
  });
}

#endif  // KALDI_PYBIND_TREE_CONTEXT_DEP_PYBIND_H_
