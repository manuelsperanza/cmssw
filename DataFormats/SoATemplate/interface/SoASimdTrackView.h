#ifndef DataFormats_SoATemplate_interface_SoASimdTrackView_h
#define DataFormats_SoATemplate_interface_SoASimdTrackView_h

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// std::experimental::simd is host-only, so it must not be seen by the CUDA/ROCm
// device pass. Everything below still compiles there, with max_lanes == 1.
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define SOA_SIMD_HOST_PATH 1
#include "DataFormats/SoATemplate/interface/SoASimdMasked.h"
#endif

// Masked-SIMD view for the hitOffsets column of reco::TrackLayout's View.
//
// WHY: the stock View range-checks every row inside operator[], and that check
// contains a throw. A potentially-throwing loop body is disqualified from auto-
// vectorisation (GCC: "not vectorized: statement can throw an exception", plus
// "statement clobbers memory" since the throw helper is a non-pure call). Here the
// check is hoisted out of the vector body and done once per chunk, before any store.
// Same guarantee, 1/width the cost, and the stores are explicit vector ops so no
// autovectoriser permission is needed.
//
// DESIGN: the CPU/GPU choice lives HERE, not in the kernel. `uniform_spans` yields
// runs of rows; on CPU backends a run is up to `width` adjacent rows, on GPU backends
// it is always a single row, so the same kernel loop compiles to the original scalar
// code on device and to masked SIMD on host.
namespace soa_simd {

#ifdef SOA_SIMD_HOST_PATH
  inline constexpr int max_lanes = width;
#else
  inline constexpr int max_lanes = 1;
#endif

  // A run of rows [index, index + lanes). Converts to int so it can still be used
  // wherever a plain row index was used before.
  struct Span {
    int index;
    int lanes;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr operator int() const { return index; }
  };

  /* uniform_spans(acc, extent)
   *
   * Same traversal as cms::alpakatools::uniform_elements, but yielding Spans instead
   * of single indices. Mirrors UniformElementsAlong: each work item takes a run of
   * `elements` rows starting at `first`, then advances by the grid `stride`.
   *
   * On CPU backends make_workdiv() gives threads/block == 1 and elements/thread ==
   * block size, so a run is a CONTIGUOUS block of rows and SIMD lanes map to adjacent
   * rows. On GPU backends elements/thread == 1, so every Span has lanes == 1 and the
   * traversal is identical to uniform_elements.
   */
  template <typename TAcc>
  class UniformSpans {
  public:
    ALPAKA_FN_ACC ALPAKA_FN_INLINE UniformSpans(TAcc const &acc, int extent)
        : extent_{extent},
          elements_{static_cast<int>(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u])},
          first_{static_cast<int>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]) *
                 static_cast<int>(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u])},
          stride_{static_cast<int>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]) *
                  static_cast<int>(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u])} {}

    class iterator {
    public:
      ALPAKA_FN_ACC ALPAKA_FN_INLINE iterator(int extent, int elements, int stride, int base)
          : extent_{extent}, elements_{elements}, stride_{stride}, base_{base} {
        index_ = base_;
        runEnd_ = base_ < extent_ ? std::min(base_ + elements_, extent_) : extent_;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE Span operator*() const {
        return Span{index_, std::min(max_lanes, runEnd_ - index_)};
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE iterator &operator++() {
        index_ += std::min(max_lanes, runEnd_ - index_);
        if (index_ < runEnd_)
          return *this;
        base_ += stride_;
        if (base_ < extent_) {
          index_ = base_;
          runEnd_ = std::min(base_ + elements_, extent_);
        } else {
          base_ = extent_;
          index_ = extent_;
          runEnd_ = extent_;
        }
        return *this;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool operator!=(iterator const &o) const { return index_ != o.index_; }

    private:
      int extent_, elements_, stride_, base_, index_, runEnd_;
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE iterator begin() const { return iterator(extent_, elements_, stride_, first_); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE iterator end() const { return iterator(extent_, elements_, stride_, extent_); }

  private:
    int extent_, elements_, first_, stride_;
  };

  template <typename TAcc, typename TExtent>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto uniform_spans(TAcc const &acc, TExtent extent) {
    return UniformSpans<TAcc>(acc, static_cast<int>(extent));
  }

  // Load `span.lanes` values starting at p. On CPU a masked vector load, on GPU a
  // plain scalar load -- identical to what the original kernel line did.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE auto load(const uint32_t *p, Span span) {
#ifdef SOA_SIMD_HOST_PATH
    if constexpr (max_lanes > 1)
      return maskedLoad(reinterpret_cast<const int *>(p), span.lanes);
    else
#endif
      return *p;
  }

  template <typename BaseView>
  struct SimdTrackView : public BaseView {
    using BaseView::BaseView;
    // Views are cheap value types (pointers + size), so wrapping by value is fine.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE SimdTrackView(const BaseView &v) : BaseView(v) {}

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int size() const { return static_cast<int>(this->metadata().size()); }

    // store proxy: ref.hitOffsets() = v  ->  masked store of the valid lanes only
    struct hitOffsets_ref {
      uint32_t *p_;
      int lanes_;
#ifdef SOA_SIMD_HOST_PATH
      ALPAKA_FN_INLINE void operator=(const vint &v) const { maskedStore(reinterpret_cast<int *>(p_), v, lanes_); }
#endif
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void operator=(uint32_t v) const { *p_ = v; }
    };

    struct element_ref {
      uint32_t *p_;
      int lanes_;
      ALPAKA_FN_ACC ALPAKA_FN_INLINE hitOffsets_ref hitOffsets() const { return {p_, lanes_}; }
    };

    // THE RANGE CHECK, batched: validated once for the whole span, before any store.
    // Unconditional on purpose -- it keeps this header compatible with both the old
    // bool-based and the new Mode-enum RangeChecking APIs, and the point of the
    // exercise is to measure the "checking ON *and* vectorised" configuration.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE element_ref operator[](Span span) const {
      const int n = std::min(span.lanes, size() - span.index);
      if (span.index < 0 || n < 0 || span.index + n > size()) {
#ifdef SOA_SIMD_HOST_PATH
        throw std::out_of_range("SimdTrackView::operator[]: span out of range");
#else
        ALPAKA_ASSERT_ACC(false);
#endif
      }
      return element_ref{const_cast<uint32_t *>(this->metadata().addressOf_hitOffsets()) + span.index, n};
    }
  };

}  // namespace soa_simd

#endif  // DataFormats_SoATemplate_interface_SoASimdTrackView_h
