#pragma once

#include <experimental/simd>
#include <stdexcept>

namespace stdx = std::experimental;

// A simd vector plus its number of valid lanes (n_ == V::size() when full).
// The whole-vector value v_ is untouched -- arithmetic still runs on all lanes
// (the trailing ones hold zero-filled garbage and are dropped by the masked
// store). n_ exists only to guard per-lane access: reading a trailing lane
// throws instead of silently handing back a bogus value.
template <typename V>
struct laned {
  V v_;
  int n_;

  auto operator[](int k) const {
    if (k < 0 || k >= n_)
      throw std::out_of_range("laned: lane out of range");
    return v_[k];
  }
  int size() const { return n_; }

  friend laned operator*(const laned &a, const laned &b) {
    return {a.v_ * b.v_, a.n_ < b.n_ ? a.n_ : b.n_};
  }
  friend laned operator+(const laned &a, const laned &b) {
    return {a.v_ + b.v_, a.n_ < b.n_ ? a.n_ : b.n_};
  }
};

template <typename BaseInputView>
struct SimdInputView : public BaseInputView {
  using BaseInputView::BaseInputView;

  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  static constexpr int width = static_cast<int>(vfloat::size());

  static vfloat maskedLoad(const float *p, int lanes) {
    vfloat v(0.f);
    if (lanes >= width) {
      v.copy_from(p, stdx::element_aligned);
      return v;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), v).copy_from(p, stdx::element_aligned);
    return v;
  }

  static vint maskedLoad(const int *p, int lanes) {
    vint v(0);
    if (lanes >= width) {
      v.copy_from(p, stdx::element_aligned);
      return v;
    }
    const vint lane([](auto k) { return static_cast<int>(k); });
    stdx::where(lane < vint(lanes), v).copy_from(p, stdx::element_aligned);
    return v;
  }

  static void maskedStore(float *p, const vfloat &v, int lanes) {
    if (lanes >= width) {
      v.copy_to(p, stdx::element_aligned);
      return;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), v).copy_to(p, stdx::element_aligned);
  }

  static void maskedStore(int *p, const vint &v, int lanes) {
    if (lanes >= width) {
      v.copy_to(p, stdx::element_aligned);
      return;
    }
    const vint lane([](auto k) { return static_cast<int>(k); });
    stdx::where(lane < vint(lanes), v).copy_to(p, stdx::element_aligned);
  }

  int lanesAt(int i) const {
    const int rem = static_cast<int>(this->metadata().size()) - i;
    return rem < width ? rem : width;
  }

  // Value element: carries the loaded vectors + the chunk's valid-lane count.
  // Accessors hand back laned<> so trailing lanes cannot be read by hand.
  struct simd_element {
    vfloat x_, y_, z_;
    vint idx_;
    int lanes_ = width;

    laned<vfloat> x() const { return {x_, lanes_}; }
    laned<vfloat> y() const { return {y_, lanes_}; }
    laned<vfloat> z() const { return {z_, lanes_}; }
    laned<vint> idx() const { return {idx_, lanes_}; }
  };

  // Write proxy: same accessors, plus store-on-assignment.
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

  // Gather by a raw vector of indices: every lane is an independent row, so the
  // result is full (lanes_ == width).
  simd_element operator[](const vint &vj) const {
    if constexpr (BaseInputView::rangeChecking == cms::soa::RangeChecking::enabled) {
      const vint n(static_cast<int>(this->metadata().size()));
      auto bad = (vj < vint(0)) || (vj >= n);
      if (stdx::any_of(bad))
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

  // Gather by laned indices: the lane count flows through, so a tail chunk's
  // gathered row is marked partial and its trailing lanes stay inaccessible.
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

  static void maskedStore(float *p, const vfloat &v, int lanes) {
    if (lanes >= width) {
      v.copy_to(p, stdx::element_aligned);
      return;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), v).copy_to(p, stdx::element_aligned);
  }

  int lanesAt(int i) const {
    const int rem = static_cast<int>(this->metadata().size()) - i;
    return rem < width ? rem : width;
  }

  // --- contiguous chunk at i: sresult[i].a() = va ---------------------------
  struct a_chunk_ref {
    float *p_;
    int lanes_;
    const a_chunk_ref &operator=(const laned<vfloat> &va) const {
      maskedStore(p_, va.v_, lanes_);
      return *this;
    }
    // read one valid lane; a trailing lane throws.
    float operator[](int k) const {
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

  // --- scatter at vj: sresult[vj].a() = va ----------------------------------
  // Lane-by-lane (no hardware scatter below AVX-512). Only the lanes_ valid
  // lanes are written, so a partial tail chunk no longer scatters into row 0.
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
      auto bad = (vj.v_ < vint(0)) || (vj.v_ >= n);
      if (stdx::any_of(bad))
        throw std::out_of_range("SimdResultView::operator[](vint) out of range");
    }
    return simd_element_scatter_ref{this->metadata().addressOf_a(), vj.v_, vj.n_};
  }
};
