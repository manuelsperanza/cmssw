#pragma once

#include <experimental/simd>
#include <stdexcept>

namespace stdx = std::experimental;

// Extends an existing range-checked SoA View with methods that return a
// std::experimental::simd object (W lanes at once) instead of a single
// element, batching the range check once per chunk instead of once per
// lane. Built entirely on the View's own public API (metadata(),
// addressOf_*(), rangeChecking) -- SoALayout.h itself is untouched.
//
// Specific to the InputLayout used in SoAIndexedAccessSolStd.cc
// (columns: x, y, z, idx) -- same scope as that file, just reusable.
template <typename BaseInputView>
struct SimdInputView : public BaseInputView {
  using BaseInputView::BaseInputView;

  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  static constexpr int width = static_cast<int>(vfloat::size());

  // SIMD analog of `element operator[](i)`: one chunk (W lanes) of x,y,z
  // instead of one scalar of each.
  struct simd_element {
    vfloat x, y, z;
  };

  // Contiguous, chunk-checked load of `width` idx values starting at i.
  // One bound check for the whole chunk (i + width <= size), instead of
  // relying on the compiler to prove W per-element checks dead.
  // always_inline so the wrapper collapses into the caller's loop even
  // without LTO (the scattered gather body is large enough that GCC's -O3
  // inliner leaves it out-of-line otherwise -> a call + stack return per chunk).
  inline __attribute__((always_inline)) vint simdLoadIdx(int i) const {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i + width > static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdInputView::simdLoadIdx out of range");
    }
    return vint(this->metadata().addressOf_idx() + i, stdx::element_aligned);
  }

  // Gather x,y,z at the `width` indices in j. All lanes are range-checked
  // as ONE vector compare + any_of before any memory is touched -- same
  // idea as the hand-written loop in SoAIndexedAccessSolStd.cc, wrapped
  // into a reusable method.
  inline __attribute__((always_inline)) simd_element simdGather(const vint& j) const {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      const vint n(static_cast<int>(this->metadata().size()));
      auto bad = (j < vint(0)) || (j >= n);
      if (stdx::any_of(bad))
        throw std::out_of_range("SimdInputView::simdGather out of range");
    }
    const float* x = this->metadata().addressOf_x();
    const float* y = this->metadata().addressOf_y();
    const float* z = this->metadata().addressOf_z();
    return simd_element{vfloat([&](auto k) { return x[j[k]]; }),
                     vfloat([&](auto k) { return y[j[k]]; }),
                     vfloat([&](auto k) { return z[j[k]]; })};
  }
};

// Extends the (single-column) result View with a chunk-checked SIMD store.
template <typename BaseResultView>
struct SimdResultView : public BaseResultView {
  using BaseResultView::BaseResultView;

  using vfloat = stdx::native_simd<float>;
  static constexpr int width = static_cast<int>(vfloat::size());

  // NOT const: writing the result buffer needs the mutable metadata()
  // overload, whose addressOf_a() returns a writable float* (the const
  // overload returns const float*, which copy_to cannot store through).
  inline __attribute__((always_inline)) void simdStoreA(int i, const vfloat& va) {
    if constexpr (BaseResultView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i + width > static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdResultView::simdStoreA out of range");
    }
    va.copy_to(this->metadata().addressOf_a() + i, stdx::element_aligned);
  }
};
