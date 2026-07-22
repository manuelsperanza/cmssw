#ifndef DataFormats_SoATemplate_test_SoASimdView_h
#define DataFormats_SoATemplate_test_SoASimdView_h

#include <experimental/simd>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoACommon.h"  // cms::soa::RangeChecking

#include "SoASimdLaned.h"
#include "SoASimdMasked.h"

// SIMD wrappers over a range-checked SoA View. They expose the same operator[]
// API as the base View but return a chunk of `width` rows at once, so a loop
// stepping by `width` keeps the exact source shape of the scalar version. The
// framework range check is preserved (batched once per chunk); the partial
// final chunk is handled by masking, and its trailing lanes are made
// inaccessible through laned<> (SoASimdLaned.h).

namespace stdx = std::experimental;
using soa_simd::maskedLoad;
using soa_simd::maskedStore;

template <typename BaseInputView>
struct SimdInputView : public BaseInputView {
  using BaseInputView::BaseInputView;

  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  static constexpr int width = static_cast<int>(vfloat::size());

  // valid lanes in the chunk starting at i (== width except for the last one)
  int lanesAt(int i) const {
    const int rem = static_cast<int>(this->metadata().size()) - i;
    return rem < width ? rem : width;
  }

  // a chunk of x,y,z,idx plus its valid-lane count; accessors return laned<> so
  // trailing lanes cannot be read by hand.
  struct simd_element {
    vfloat x_, y_, z_;
    vint idx_;
    int lanes_ = width;

    laned<vfloat> x() const { return {x_, lanes_}; }
    laned<vfloat> y() const { return {y_, lanes_}; }
    laned<vfloat> z() const { return {z_, lanes_}; }
    laned<vint> idx() const { return {idx_, lanes_}; }
  };

  // write proxy: same accessors, plus masked store on assignment
  struct simd_element_ref {
    float *x_, *y_, *z_;
    int *idx_;
    int lanes_;

    simd_element_ref &operator=(const simd_element &v) {
      maskedStore(x_, v.x_, lanes_);
      maskedStore(y_, v.y_, lanes_);
      maskedStore(z_, v.z_, lanes_);
      maskedStore(idx_, v.idx_, lanes_);
      return *this;
    }

    laned<vfloat> x() const { return {maskedLoad(x_, lanes_), lanes_}; }
    laned<vfloat> y() const { return {maskedLoad(y_, lanes_), lanes_}; }
    laned<vfloat> z() const { return {maskedLoad(z_, lanes_), lanes_}; }
    laned<vint> idx() const { return {maskedLoad(idx_, lanes_), lanes_}; }
  };

  // non-const: write the chunk at i
  simd_element_ref operator[](int i) {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i >= static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdInputView::operator[] out of range");
    }
    return simd_element_ref{this->metadata().addressOf_x() + i,
                            this->metadata().addressOf_y() + i,
                            this->metadata().addressOf_z() + i,
                            this->metadata().addressOf_idx() + i,
                            lanesAt(i)};
  }

  // const: read the contiguous chunk at i
  simd_element operator[](int i) const {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i >= static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdInputView::operator[] out of range");
    }
    const int lanes = lanesAt(i);
    return simd_element{maskedLoad(this->metadata().addressOf_x() + i, lanes),
                        maskedLoad(this->metadata().addressOf_y() + i, lanes),
                        maskedLoad(this->metadata().addressOf_z() + i, lanes),
                        maskedLoad(this->metadata().addressOf_idx() + i, lanes),
                        lanes};
  }

  // gather at an explicit vector of rows; every lane is an independent full row
  simd_element operator[](const vint &vj) const {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      const vint n(static_cast<int>(this->metadata().size()));
      if (stdx::any_of((vj < vint(0)) || (vj >= n)))
        throw std::out_of_range("SimdInputView::operator[](vint) out of range");
    }
    const float *x = this->metadata().addressOf_x();
    const float *y = this->metadata().addressOf_y();
    const float *z = this->metadata().addressOf_z();
    const int *idx = this->metadata().addressOf_idx();
    return simd_element{vfloat([&](auto k) { return x[vj[k]]; }),
                        vfloat([&](auto k) { return y[vj[k]]; }),
                        vfloat([&](auto k) { return z[vj[k]]; }),
                        vint([&](auto k) { return idx[vj[k]]; }),
                        width};
  }

  // gather at laned indices: the lane count flows through, so a tail chunk's
  // gathered row keeps its trailing lanes inaccessible
  simd_element operator[](const laned<vint> &vj) const {
    simd_element e = (*this)[vj.v_];
    e.lanes_ = vj.n_;
    return e;
  }
};

template <typename BaseResultView>
struct SimdResultView : public BaseResultView {
  using BaseResultView::BaseResultView;

  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  static constexpr int width = static_cast<int>(vfloat::size());

  int lanesAt(int i) const {
    const int rem = static_cast<int>(this->metadata().size()) - i;
    return rem < width ? rem : width;
  }

  // contiguous store: sresult[i].a() = va
  struct a_chunk_ref {
    float *p_;
    int lanes_;
    const a_chunk_ref &operator=(const laned<vfloat> &va) const {
      maskedStore(p_, va.v_, lanes_);
      return *this;
    }
    float operator[](int k) const {  // read one valid lane; a trailing lane throws
      if (k < 0 || k >= lanes_)
        throw std::out_of_range("a_chunk_ref: lane out of range");
      return p_[k];
    }
  };

  struct simd_element_ref {
    float *p_;
    int lanes_;
    a_chunk_ref a() const { return a_chunk_ref{p_, lanes_}; }
  };

  simd_element_ref operator[](int i) {
    if constexpr (BaseResultView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i >= static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdResultView::operator[] out of range");
    }
    return simd_element_ref{this->metadata().addressOf_a() + i, lanesAt(i)};
  }

  // scatter store: sresult[vj].a() = va. No hardware scatter below AVX-512, so
  // lane by lane -- only the valid lanes are written, so a partial tail chunk no
  // longer scatters into row 0.
  struct a_scatter_ref {
    float *base_;
    vint vj_;
    int lanes_;
    const a_scatter_ref &operator=(const laned<vfloat> &va) const {
      for (int k = 0; k < lanes_; ++k)
        base_[vj_[k]] = va.v_[k];
      return *this;
    }
  };

  struct simd_element_scatter_ref {
    float *base_;
    vint vj_;
    int lanes_;
    a_scatter_ref a() const { return a_scatter_ref{base_, vj_, lanes_}; }
  };

  simd_element_scatter_ref operator[](const laned<vint> &vj) {
    if constexpr (BaseResultView::rangeChecking == cms::soa::RangeChecking::enabled) {
      const vint n(static_cast<int>(this->metadata().size()));
      if (stdx::any_of((vj.v_ < vint(0)) || (vj.v_ >= n)))
        throw std::out_of_range("SimdResultView::operator[](vint) out of range");
    }
    return simd_element_scatter_ref{this->metadata().addressOf_a(), vj.v_, vj.n_};
  }
};

#endif  // DataFormats_SoATemplate_test_SoASimdView_h
