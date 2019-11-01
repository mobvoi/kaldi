// chain/chain-den-graph.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


// See ../../COPYING for clarification regarding multiple authors
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


#ifndef KALDI_CHAIN_CHAIN_DEN_GRAPH_H_
#define KALDI_CHAIN_CHAIN_DEN_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "chain/chain-datastruct.h"
#include "hmm/transition-model.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace chain {


/**  This class is responsible for storing the FST that we use as the
     'anti-model' or 'denominator-model', that models all possible phone
     sequences (or most possible phone sequences, depending how we built it)..
     It stores the FST in a format where we can access both the transitions out
     of each state, and the transitions into each state.

     This class supports both GPU and non-GPU operation, but is optimized for
     GPU.
 */
class DenominatorGraph {
 public:

  // the number of states in the HMM.
  int32 NumStates() const;

  // the number of PDFs (the labels on the transitions are numbered from 0 to
  // NumPdfs() - 1).
  int32 NumPdfs() const { return num_pdfs_; }

  DenominatorGraph();

  // Initialize from epsilon-free acceptor FST with pdf-ids plus one as the
  // labels.  'num_pdfs' is only needeed for checking.
  DenominatorGraph(const fst::StdVectorFst &fst,
                   int32 num_pdfs);

  // returns the pointer to the forward-transitions array, indexed by hmm-state,
  // which will be on the GPU if we're using a GPU.
  const Int32Pair *ForwardTransitions() const;

  // returns the pointer to the backward-transitions array, indexed by
  // hmm-state, which will be on the GPU if we're using a GPU.
  const Int32Pair *BackwardTransitions() const;

  // returns the array to the actual transitions (this is indexed by the ranges
  // returned from the ForwardTransitions and BackwardTransitions arrays).  The
  // memory will be GPU memory if we are using a GPU.
  const DenominatorGraphTransition *Transitions() const;


  // returns the cold-start versions of the initial-probs of the HMM-states, for
  // the FST passed to the constructor, stored as real probabilities, not in
  // log-space.  This will actually a vector that's one for a particular state
  // and zero elsewhere, since FSTs have just one initial state...
  const CuVector<BaseFloat> &RealInitialProbs() const {
    return real_initial_probs_;
  }

  // returns the warm-start (i.e. cut-point) versions of the initial-probs of
  // the HMM-states (as real probabilities, not in log-space).  these are used
  // when we are entering in the middle of a sequence.  They are approximate
  // initial-probs obtained by running the HMM for a fixed number of time-steps
  // (e.g. 100) and averaging the posteriors over those time-steps.
  const CuVector<BaseFloat> &SplitPointInitialProbs() const {
    return split_point_initial_probs_;
  }


  // Returns the final-probs of the HMM-states... these are only used when
  // we are genuinely at the end of a sequence (not just at the end of a chunk).
  // At the end of a chunk we'd use a vector of all ones.
  const CuVector<BaseFloat> &RealFinalProbs() const {
    return real_final_probs_;
  }

  /// This returns a vector of ones; it is used for symmetry with what happens
  /// at the start.  (If we are at a split point we treat all states as final
  /// with probability one).
  const CuVector<BaseFloat> &SplitPointFinalProbs() const {
    return split_point_final_probs_;
  }

  // This function outputs a modified version of the FST that was used to
  // build this object, that has an initial-state with epsilon transitions to
  // each state, with weight determined by SplitPointInitialProbs(); and has each original
  // state being final with probability one (note: we remove epsilons).  This is
  // used in computing the 'penalty_logprob' of the Supervision objects, to
  // ensure that the objective function is never positive, which makes it more
  // easily interpretable.  'ifst' must be the same FST that was provided to the
  // constructor of this object.  [note: ifst and ofst may be the same object.]
  // This function ensures that 'ofst' is ilabel sorted (which will be useful in
  // composition).
  //
  // CAUTION: this has become a little inexact/suboptimal now that we are
  // distinguishing how the den-graph starts and terminates depending on
  // whether we were at the end of a chunk or not.  The normalization FST
  // only gives correct probs for interior chunks.  This only affects
  // diagnostics, though.
  void GetNormalizationFst(const fst::StdVectorFst &ifst,
                           fst::StdVectorFst *ofst);

  // Use default copy constructor and assignment operator.
 private:
  // functions called from the constructor
  void SetTransitions(const fst::StdVectorFst &fst, int32 num_pfds);


  // work out the initial and final probability vectors real_initial_probs_
  // through split_point_final_probs_.
  void SetEdgeProbs(const fst::StdVectorFst &fst);

  // forward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> forward_transitions_;
  // backward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> backward_transitions_;
  // This stores the actual transitions.
  CuArray<DenominatorGraphTransition> transitions_;

  // The initial-probability of each state in the den-graph, used on the first
  // frame of a sequence.  These are the real ones from the original compiled
  // denominator graph-- for use when it's truly at the start of a sequence.
  // This will actually be a zero-one vector.
  CuVector<BaseFloat> real_initial_probs_;

  // The initial-probs used for each state when a chunk starts in the middle of
  // an utterance.  These are derived from the average occupation-prob, in the
  // denominator FST, of each FST state.
  CuVector<BaseFloat> split_point_initial_probs_;

  // The final-probs of each state within the original compiled denominator
  // graph.  These are used when the end of a chunk occurs at the end of
  // an utterance.
  CuVector<BaseFloat> real_final_probs_;

  // These are "fake" final-probs for use when a chunk ends within an
  // utterance.  They are all ones.
  CuVector<BaseFloat> split_point_final_probs_;


  int32 num_pdfs_;
};


// Function that does acceptor minimization without weight pushing...
// this is useful when constructing the denominator graph.
void MinimizeAcceptorNoPush(fst::StdVectorFst *fst);

// Utility function used while building the graph.  Converts
// transition-ids to pdf-ids plus one.  Assumes 'fst'
// is an acceptor, but does not check this (only looks at its
// ilabels).
void MapFstToPdfIdsPlusOne(const TransitionModel &trans_model,
                           fst::StdVectorFst *fst);

// Starting from an acceptor on phones that represents some kind of compiled
// language model (with no disambiguation symbols), this funtion creates the
// denominator-graph.  Note: there is similar code in chain-supervision.cc, when
// creating the supervision graph.
void CreateDenominatorFst(const ContextDependency &ctx_dep,
                          const TransitionModel &trans_model,
                          const fst::StdVectorFst &phone_lm,
                          fst::StdVectorFst *den_graph);


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_DEN_GRAPH_H_
