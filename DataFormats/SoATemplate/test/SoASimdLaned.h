#ifndef DataFormats_SoATemplate_test_SoASimdLaned_h
#define DataFormats_SoATemplate_test_SoASimdLaned_h

#include <stdexcept>

// A SIMD vector v_ plus its number of valid lanes n_ (== the full width when the
// chunk is complete). Whole-vector arithmetic still runs on every lane; n_ only
// guards per-lane access, so reading a trailing (padding) lane throws instead of
// silently returning a bogus value.
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

#endif  // DataFormats_SoATemplate_test_SoASimdLaned_h
