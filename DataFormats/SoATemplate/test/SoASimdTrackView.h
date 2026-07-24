#ifndef DataFormats_SoATemplate_test_SoASimdTrackView_h
#define DataFormats_SoATemplate_test_SoASimdTrackView_h

#include <cstdint>
#include <experimental/simd>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoACommon.h"  // cms::soa::RangeChecking

#include "SoASimdMasked.h"

// Experimental SIMD result-view for the hitOffsets column of reco::TrackLayout's
// View. Same idea as SimdResultView in SoASimdView.h (which was written for the
// x/y/z/a test layout): wrap a range-checked SoA View, expose the same operator[]
// shape but return a chunk of `width` rows at once. Assigning a masked vector to
// hitOffsets() stores only the valid lanes, and the framework range check is
// preserved -- batched once per chunk instead of once per row.
namespace soa_simd {

  template <typename BaseView>
  struct SimdTrackView : public BaseView {
    using BaseView::BaseView;
    // wrap an existing View (Views are cheap value types: pointers + size)
    SimdTrackView(const BaseView &v) : BaseView(v) {}

    // valid lanes in the chunk starting at i (== width except for the last one)
    int lanesAt(int i) const {
      const int rem = static_cast<int>(this->metadata().size()) - i;
      return rem < width ? rem : width;
    }

    // store proxy: sview[i].hitOffsets() = v  -> masked store of the valid lanes
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

    element_ref operator[](int i) {
      const int lanes = lanesAt(i);
      if constexpr (BaseView::rangeChecking == cms::soa::RangeChecking::enabled) {
        if (i < 0 || i + lanes > static_cast<int>(this->metadata().size()))
          throw std::out_of_range("SimdTrackView::operator[] out of range");
      }
      return element_ref{this->metadata().addressOf_hitOffsets() + i, lanes};
    }
  };

}  // namespace soa_simd

#endif  // DataFormats_SoATemplate_test_SoASimdTrackView_h
