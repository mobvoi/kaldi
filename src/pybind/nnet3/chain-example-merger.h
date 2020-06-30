// pybind/nnet3/chain-example-merger.h

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

#ifndef KALDI_PYBIND_NNET3_CHAIN_EXAMPLE_MERGER_H_
#define KALDI_PYBIND_NNET3_CHAIN_EXAMPLE_MERGER_H_

#include <deque>

#include "nnet3/nnet-chain-example.h"

namespace kaldi {
namespace nnet3 {

// Note(fangjun): this class is modified from nnet3/nnet-chain-example.cc.
// Since we do not want to write the merged egs to files, we create
// this class to keep the merged egs in memory, which will be consumed
// by PyTorch.
class ChainExampleMerger2 {
 public:
  explicit ChainExampleMerger2(const ExampleMergingConfig& config);

  // This function accepts an example, and if possible, writes a merged example
  // out.  The ownership of the pointer 'a' is transferred to this class when
  // you call this function.
  void AcceptExample(NnetChainExample* a);

  // This function announces to the class that the input has finished, so it
  // should flush out any smaller-sized minibatches, as dictated by the config.
  // This will be called in the destructor, but you can call it explicitly when
  // all the input is done if you want to; it won't repeat anything if called
  // twice.  It also prints the stats.
  void Finish();

  // returns a suitable exit status for a program.
  int32 ExitStatus() {
    Finish();
    return (num_egs_written_ > 0 ? 0 : 1);
  }

  ~ChainExampleMerger2() { Finish(); };

  int Size() const { return cegs_.size(); }
  // TODO(fangjun): support move semantics
  std::pair<std::string, NnetChainExample> Get() const { return cegs_.front(); }
  void Pop() { cegs_.pop_front(); }

 private:
  // called by Finish() and AcceptExample().  Merges, updates the stats, and
  // writes.  The 'egs' is non-const only because the egs are temporarily
  // changed inside MergeChainEgs.  The pointer 'egs' is still owned
  // by the caller.
  void WriteMinibatch(std::vector<NnetChainExample>* egs);

  bool finished_;
  int32 num_egs_written_;
  const ExampleMergingConfig& config_;

  // Note: the "key" into the egs is the first element of the vector.
  typedef unordered_map<NnetChainExample*, std::vector<NnetChainExample*>,
                        NnetChainExampleStructureHasher,
                        NnetChainExampleStructureCompare>
      MapType;
  MapType eg_to_egs_;
  std::deque<std::pair<std::string, NnetChainExample>> cegs_;
};

}  // namespace nnet3
}  // namespace kaldi

#endif  // KALDI_PYBIND_NNET3_CHAIN_EXAMPLE_MERGER_H_
