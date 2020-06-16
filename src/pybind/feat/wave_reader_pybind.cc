// pybind/feat/wave_reader_pybind.cc

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

#include "feat/wave_reader_pybind.h"

#include "feat/wave-reader.h"
#include "util/kaldi_table_pybind.h"

using namespace kaldi;

static void pybind_wave_info(py::module& m) {
  m.attr("kWaveSampleMax") = py::cast(kWaveSampleMax);
  using PyClass = WaveInfo;
  DEF_CLASS("WaveInfo");
  DEF_INIT();
  DEF(IsStreamed);
  DEF(SampFreq);
  DEF(SampleCount);
  DEF(Duration);
  DEF(NumChannels);
  DEF(BlockAlign);
  DEF(DataBytes);
  DEF(ReverseBytes);
}

static void pybind_wave_data(py::module& m) {
  using PyClass = WaveData;
  DEF_CLASS("WaveData");
  DEF_INIT();
  pyclass.def(py::init<const float, const Matrix<float>>(),
              py::arg("samp_freq"), py::arg("data"));
  DEF_REF(Data);
  DEF(SampFreq);
  DEF(Duration);
  DEF(Clear);
}

void pybind_wave_reader(py::module& m) {
  pybind_wave_info(m);
  pybind_wave_data(m);

  pybind_sequential_table_reader<WaveHolder>(m, "_SequentialWaveReader");
  pybind_sequential_table_reader<WaveInfoHolder>(m,
                                                 "_SequentialWaveInfoReader");
  pybind_random_access_table_reader<WaveHolder>(m, "_RandomAccessWaveReader");
  pybind_random_access_table_reader<WaveInfoHolder>(
      m, "_RandomAccessWaveInfoReader");
}
