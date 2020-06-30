// pybind/nnet3/chain-example-merger.cc

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

#include "nnet3/chain-example-merger.h"

namespace kaldi {
namespace nnet3 {

extern int32 GetNnetChainExampleSize(const NnetChainExample& a);

ChainExampleMerger2::ChainExampleMerger2(const ExampleMergingConfig& config)
    : finished_(false), num_egs_written_(0), config_(config) {}

void ChainExampleMerger2::AcceptExample(NnetChainExample* eg) {
  KALDI_ASSERT(!finished_);
  // If an eg with the same structure as 'eg' is already a key in the
  // map, it won't be replaced, but if it's new it will be made
  // the key.  Also we remove the key before making the vector empty.
  // This way we ensure that the eg in the key is always the first
  // element of the vector.
  std::vector<NnetChainExample*>& vec = eg_to_egs_[eg];
  vec.push_back(eg);
  int32 eg_size = GetNnetChainExampleSize(*eg), num_available = vec.size();
  bool input_ended = false;
  int32 minibatch_size =
      config_.MinibatchSize(eg_size, num_available, input_ended);
  if (minibatch_size != 0) {  // we need to write out a merged eg.
    KALDI_ASSERT(minibatch_size == num_available);

    std::vector<NnetChainExample*> vec_copy(vec);
    eg_to_egs_.erase(eg);

    // MergeChainExamples() expects a vector of NnetChainExample, not of
    // pointers, so use swap to create that without doing any real work.
    std::vector<NnetChainExample> egs_to_merge(minibatch_size);
    for (int32 i = 0; i < minibatch_size; i++) {
      egs_to_merge[i].Swap(vec_copy[i]);
      delete vec_copy[i];  // we owned those pointers.
    }
    WriteMinibatch(&egs_to_merge);
  }
}

void ChainExampleMerger2::WriteMinibatch(std::vector<NnetChainExample>* egs) {
  KALDI_ASSERT(!egs->empty());

  int32 minibatch_size = egs->size();

  NnetChainExample merged_eg;
  MergeChainExamples(config_.compress, egs, &merged_eg);
  std::ostringstream key;
  std::string suffix = "";
  if (config_.multilingual_eg) {
    // pick the first output's suffix
    std::string output_name = merged_eg.outputs[0].name;
    const size_t pos = output_name.find('-');
    const size_t len = output_name.length();
    suffix = "?lang=" + output_name.substr(pos + 1, len);
  }
  key << "merged-" << (num_egs_written_++) << "-" << minibatch_size << suffix;
  // TODO(fangjun): support move semantics of NnetChainExample
  cegs_.emplace_back(key.str(), merged_eg);
}

void ChainExampleMerger2::Finish() {
  if (finished_) return;  // already finished.
  finished_ = true;

  // we'll convert the map eg_to_egs_ to a vector of vectors to avoid
  // iterator invalidation problems.
  std::vector<std::vector<NnetChainExample*> > all_egs;
  all_egs.reserve(eg_to_egs_.size());

  MapType::iterator iter = eg_to_egs_.begin(), end = eg_to_egs_.end();
  for (; iter != end; ++iter) all_egs.push_back(iter->second);
  eg_to_egs_.clear();

  for (size_t i = 0; i < all_egs.size(); i++) {
    int32 minibatch_size;
    std::vector<NnetChainExample*>& vec = all_egs[i];
    KALDI_ASSERT(!vec.empty());
    int32 eg_size = GetNnetChainExampleSize(*(vec[0]));
    bool input_ended = true;
    while (!vec.empty() && (minibatch_size = config_.MinibatchSize(
                                eg_size, vec.size(), input_ended)) != 0) {
      // MergeChainExamples() expects a vector of
      // NnetChainExample, not of pointers, so use swap to create that
      // without doing any real work.
      std::vector<NnetChainExample> egs_to_merge(minibatch_size);
      for (int32 i = 0; i < minibatch_size; i++) {
        egs_to_merge[i].Swap(vec[i]);
        delete vec[i];  // we owned those pointers.
      }
      vec.erase(vec.begin(), vec.begin() + minibatch_size);
      WriteMinibatch(&egs_to_merge);
    }
    if (!vec.empty()) {
      int32 num_discarded = vec.size();
      for (int32 i = 0; i < num_discarded; i++) delete vec[i];
      vec.clear();
    }
  }
}

}  // namespace nnet3
}  // namespace kaldi
