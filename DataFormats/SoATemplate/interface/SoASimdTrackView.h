#ifndef DataFormats_SoATemplate_interface_SoASimdTrackView_h
#define DataFormats_SoATemplate_interface_SoASimdTrackView_h

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// std::experimental::simd is host-only. __HIP_DEVICE_COMPILE__ / __CUDA_ARCH__ are NOT
// enough on their own to keep it out of device code: ALPAKA_FN_ACC expands to
// __host__ __device__, and clang type-checks such functions for BOTH targets, so a
// host-only call inside one is an error even during the host pass. The actual
// mechanism that keeps device code clean is the `if constexpr` on a TEMPLATE
// PARAMETER in copyHitOffsets() below -- a discarded branch of a template is never
// instantiated. The macro guard here only avoids *including* <experimental/simd>
// during the device pass.
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define SOA_SIMD_HOST_PATH 1
#include <stdexcept>
#include "DataFormats/SoATemplate/interface/SoASimdMasked.h"
#endif

// Masked-SIMD view for the hitOffsets column of reco::TrackLayout's View.
//
// WHY: the stock View range-checks every row inside operator[], and that check
// contains a throw. A potentially-throwing loop body is disqualified from auto-
// vectorisation (GCC: "not vectorized: statement can throw an exception", plus
// "statement clobbers memory" since the throw helper is a non-pure call). Here the
// check is hoisted out of the vector body and done once per chunk, before any store.
// Same guarantee, 1/width the cost, and the stores are explicit vector ops, so no
// autovectoriser permission is needed.
//
// DESIGN: the CPU/GPU choice lives HERE, not in the kernel. The kernel calls one
// function; CPU backends get masked SIMD, GPU backends get the original scalar loop.
namespace soa_simd {

  namespace detail {

#ifdef SOA_SIMD_HOST_PATH
    // Host-only: wraps the View and hands out a chunk of rows after ONE bounds check.
    template <typename BaseView>
    struct SimdTrackView : public BaseView {
      using BaseView::BaseView;
      // Views are cheap value types (pointers + size), so wrapping by value is fine.
      SimdTrackView(const BaseView &v) : BaseView(v) {}

      int size() const { return static_cast<int>(this->metadata().size()); }

      // store proxy: ref.hitOffsets() = v  ->  masked store of the valid lanes only
      struct hitOffsets_ref {
        uint32_t *p_;
        int lanes_;
        void operator=(const vint &v) const { maskedStore(reinterpret_cast<int *>(p_), v, lanes_); }
      };

      struct element_ref {
        uint32_t *p_;
        int lanes_;
        hitOffsets_ref hitOffsets() const { return {p_, lanes_}; }
      };

      // THE RANGE CHECK, batched: validated once for the whole chunk, before any store.
      // Unconditional on purpose -- it keeps this header compatible with both the old
      // bool-based and the new Mode-enum RangeChecking APIs, and the point of the
      // exercise is to measure the "checking ON *and* vectorised" configuration.
      element_ref chunk(int i, int lanes) const {
        const int n = std::min(lanes, size() - i);
        if (i < 0 || n < 0 || i + n > size())
          throw std::out_of_range("SimdTrackView::chunk: out of range");
        return element_ref{const_cast<uint32_t *>(this->metadata().addressOf_hitOffsets()) + i, n};
      }
    };

    // Reproduces UniformElementsAlong's traversal (workdivision.h): each work item takes
    // a run of `elements` rows starting at `first`, then advances by the grid `stride`.
    // On CPU backends that run is CONTIGUOUS, so SIMD lanes map to adjacent rows.
    template <typename TView, typename TCounter>
    inline void copyHitOffsetsSimd(
        TView tracks_view, TCounter const *off, int first, int stride, int elements, int n) {
      SimdTrackView<TView> sview{tracks_view};
      for (int base = first; base < n; base += stride) {
        const int last = std::min(base + elements, n);
        for (int i = base; i < last; i += width) {
          const int lanes = std::min(width, last - i);
          sview.chunk(i, lanes).hitOffsets() = maskedLoad(reinterpret_cast<const int *>(off + i + 1), lanes);
        }
      }
    }
#else
    // Device pass: never called (the branch below is discarded), but the NAME must be
    // declared so the discarded statement still parses.
    template <typename TView, typename TCounter>
    inline void copyHitOffsetsSimd(TView, TCounter const *, int, int, int, int) {}
#endif

  }  // namespace detail

  /* copyHitOffsets(acc, tracks_view, off, n)
   *
   * Replaces:
   *     for (auto idx : cms::alpakatools::uniform_elements(acc, n))
   *       tracks_view[idx].hitOffsets() = off[idx + 1];
   *
   * CPU backends -> masked SIMD with a batched bounds check.
   * GPU backends -> exactly the loop above, unchanged.
   *
   * The `if constexpr` condition depends on the template parameter TAcc, so the
   * host-only branch is NOT instantiated when TAcc is a GPU accelerator. That is what
   * keeps std::experimental::simd and `throw` out of device code -- preprocessor
   * guards alone cannot, because ALPAKA_FN_ACC means __host__ __device__.
   */
  template <typename TAcc, typename TView, typename TCounter>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void copyHitOffsets(TAcc const &acc, TView tracks_view, TCounter const *off, int n) {
    if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
      const int elements = static_cast<int>(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
      const int first = static_cast<int>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]) * elements;
      const int stride = static_cast<int>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]) * elements;
      detail::copyHitOffsetsSimd(tracks_view, off, first, stride, elements, n);
    } else {
      for (auto idx : cms::alpakatools::uniform_elements(acc, n)) {
        tracks_view[idx].hitOffsets() = off[idx + 1];  // offset for track 0 is always 0
      }
    }
  }

}  // namespace soa_simd

#endif  // DataFormats_SoATemplate_interface_SoASimdTrackView_h
