// pybind/nnet3/nnet_chain_example_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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

#include "nnet3/nnet_chain_example_pybind.h"

#include "nnet3/chain-example-merger.h"
#include "nnet3/nnet-chain-example.h"
#include "util/kaldi_table_pybind.h"

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace kaldi::chain;

void pybind_chain_example_merger(py::module& m) {
  using PyClass = ChainExampleMerger2;
  DEF_CLASS("ChainExampleMerger");
  pyclass.def(py::init<const ExampleMergingConfig&>(), py::arg("config"));
  pyclass.def(
      "AcceptExample", [](PyClass* self, const NnetChainExample& in_eg) {
        auto eg = new NnetChainExample(in_eg);
        self->AcceptExample(eg);  // WARNING(fangjun): self takes the ownership
      });
  DEF(Finish);
  DEF(ExitStatus);
  DEF(Size);
  DEF(Get);
  DEF(Pop);
}

static void pybind_nnet_chain_supervision(py::module& m) {
  using PyClass = NnetChainSupervision;
  DEF_CLASS("NnetChainSupervision");
  DEF_INIT();
  pyclass.def(py::init<const std::string&, const Supervision&,
                       const VectorBase<BaseFloat>&, int32, int32>(),
              py::arg("name"), py::arg("supervision"), py::arg("deriv_weights"),
              py::arg("first_frame"), py::arg("frame_skip"));
  DEF_P(name);
  DEF_P(indexes);
  DEF_P(supervision);
  DEF_P(deriv_weights);
  DEF(CheckDim);

  pyclass
      .def("__str__",
           [](const PyClass& sup) {
             std::ostringstream os;
             sup.Write(os, false);
             return os.str();
           })
      .def("ToString",
           [](const PyClass& sup) {
             std::ostringstream os;
             sup.Write(os, false);
             return os.str();
           })

      // TODO(fangjun): other methods can be wrapped when needed
      ;
}

static void pybind_vector_nnet_chain_supervision(py::module& m) {
  using PyClass = std::vector<NnetChainSupervision>;
  DEF_CLASS("NnetChainSupervisionVector");
  DEF_INIT();
  pyclass.def("resize", [](PyClass* self, size_t sz) { self->resize(sz); });
  pyclass.def("__len__", [](const PyClass& self) { return self.size(); });
  pyclass.def("__getitem__",
              [](const PyClass& self, size_t i) { return self[i]; });
  pyclass.def("__setitem__",
              [](PyClass& self, size_t i, const NnetChainSupervision& value) {
                self[i] = value;
              });
}

static void pybind_nnet_chain_example_impl(py::module& m) {
  using PyClass = NnetChainExample;
  DEF_CLASS("NnetChainExample");
  DEF_INIT();
  DEF_P(inputs);
  DEF_P(outputs);
  DEF(Compress);

  pyclass
      .def("__eq__", [](const PyClass& a, const PyClass& b) { return a == b; })
      .def("Read", &PyClass::Read, py::arg("is"), py::arg("binary"))
      .def("ToString",
           [](const PyClass& self) {
             std::ostringstream os;
             self.Write(os, false);
             return os.str();
           })
      .def("__str__", [](const PyClass& self) {
        std::ostringstream os;
        self.Write(os, false);
        return os.str();
      });

  // (fangjun): we follow the PyKaldi style to prepend a underline before the
  // registered classes and the user in general should not use them directly;
  // instead, they should use the corresponding python classes that are more
  // easier to use.
  pybind_sequential_table_reader<KaldiObjectHolder<PyClass>>(
      m, "_SequentialNnetChainExampleReader");

  pybind_random_access_table_reader<KaldiObjectHolder<PyClass>>(
      m, "_RandomAccessNnetChainExampleReader");

  pybind_table_writer<KaldiObjectHolder<PyClass>>(m, "_NnetChainExampleWriter");
}

void pybind_nnet_chain_example(py::module& m) {
  pybind_nnet_chain_supervision(m);
  pybind_vector_nnet_chain_supervision(m);
  pybind_nnet_chain_example_impl(m);
}
