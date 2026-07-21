#pragma once

#include <experimental/simd>
#include <stdexcept>

namespace stdx = std::experimental;

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

  struct simd_element {
    vfloat x_, y_, z_;
    vint idx_;

    vfloat x() const { return x_; }
    vfloat y() const { return y_; }
    vfloat z() const { return z_; }
    vint idx() const { return idx_; }
  };

  struct simd_element_ref {
    float *x_, *y_, *z_;
    int *idx_;
    int lanes_;

    simd_element_ref &operator=(const simd_element &v) {
      maskedStore(x_, v.x(), lanes_);
      maskedStore(y_, v.y(), lanes_);
      maskedStore(z_, v.z(), lanes_);
      maskedStore(idx_, v.idx(), lanes_);
      return *this;
    }

    vfloat x() const { return maskedLoad(x_, lanes_); }
    vfloat y() const { return maskedLoad(y_, lanes_); }
    vfloat z() const { return maskedLoad(z_, lanes_); }
    vint idx() const { return maskedLoad(idx_, lanes_); }
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
                        maskedLoad(this->metadata().addressOf_idx() + i, lanes)};
  }


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
                        vint([&](auto k) { return idx[vj[k]]; })};
  }
};

template <typename BaseResultView>
struct SimdResultView : public BaseResultView {
  using BaseResultView::BaseResultView;

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
  // Two-level proxy, mirroring the base View: operator[] yields an element
  // proxy, and .a() yields a column proxy that stores on assignment.
  struct a_chunk_ref {
    float *p_;
    int lanes_;
    const a_chunk_ref &operator=(const vfloat &va) const {
      maskedStore(p_, va, lanes_);
      return *this;
    }
    operator vfloat() const { return maskedLoad(p_, lanes_); }
    // Single-lane read, so `sresult[i].a()[k]` works. Lanes past the valid
    // count read as 0, matching maskedLoad.
    float operator[](int k) const { return (k >= 0 && k < lanes_) ? p_[k] : 0.f; }
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
  // x86 has no scatter instruction below AVX-512, and std::simd exposes none,
  // so this stores lane by lane. WARNING: it writes ALL width lanes -- vj must
  // hold width valid rows. On a partial final chunk the zero-filled tail lanes
  // would scatter junk into row 0, so a scattering loop must run only over full
  // chunks (i + width <= n) and finish with a scalar tail.
  struct a_scatter_ref {
    float *base_;
    vint vj_;
    const a_scatter_ref &operator=(const vfloat &va) const {
      for (std::size_t k = 0; k < vfloat::size(); ++k)
        base_[vj_[k]] = va[k];
      return *this;
    }
    operator vfloat() const {
      return vfloat([&](auto k) { return base_[vj_[k]]; });
    }
    // Single-lane read at the scattered row vj_[k].
    float operator[](int k) const { return base_[vj_[k]]; }
  };

  struct simd_element_scatter_ref {
    float *base_;
    vint vj_;
    a_scatter_ref a() const { return a_scatter_ref{base_, vj_}; }
  };

  simd_element_scatter_ref operator[](const vint &vj) {
    if constexpr (BaseResultView::rangeChecking == cms::soa::RangeChecking::enabled) {
      const vint n(static_cast<int>(this->metadata().size()));
      auto bad = (vj < vint(0)) || (vj >= n);
      if (stdx::any_of(bad))
        throw std::out_of_range("SimdResultView::operator[](vint) out of range");
    }
    return simd_element_scatter_ref{this->metadata().addressOf_a(), vj};
  }

  void simdStoreA(int i, const vfloat &va) {
    if constexpr (BaseResultView::rangeChecking == cms::soa::RangeChecking::enabled) {
      if (i < 0 || i >= static_cast<int>(this->metadata().size()))
        throw std::out_of_range("SimdResultView::simdStoreA out of range");
    }
    const int rem = static_cast<int>(this->metadata().size()) - i;
    const int lanes = rem < width ? rem : width;
    float *p = this->metadata().addressOf_a() + i;
    if (lanes >= width) {
      va.copy_to(p, stdx::element_aligned);
      return;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), va).copy_to(p, stdx::element_aligned);
  }
};
