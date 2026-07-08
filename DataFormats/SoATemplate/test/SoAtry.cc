#include <cstdlib>
#include <iostream>
#include <memory>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(InputLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))
GENERATE_SOA_LAYOUT(ResultLayoutTemplate, SOA_COLUMN(float, a))

using InputLayout = InputLayoutTemplate<>;
using ResultLayout = ResultLayoutTemplate<>;

int main() {
  constexpr int size = 128;

  std::unique_ptr<std::byte, decltype(std::free) *> inputMem{
      reinterpret_cast<std::byte *>(aligned_alloc(InputLayout::alignment, InputLayout::computeDataSize(size))),
      std::free};
  std::unique_ptr<std::byte, decltype(std::free) *> resultMem{
      reinterpret_cast<std::byte *>(aligned_alloc(ResultLayout::alignment, ResultLayout::computeDataSize(size))),
      std::free};

  InputLayout input{inputMem.get(), size};
  ResultLayout result{resultMem.get(), size};

  InputLayout::View inputView{input};
  ResultLayout::View resultView{result};

  for (int i = 0; i < size; ++i) {
    inputView[i] = {float(i), float(i) * 2.f, float(i) * 3.f};
    resultView[i].a() = inputView[i].y() * inputView[i].x() + inputView[i].z();
  }

  for (int i = 0; i < size; ++i) {
    std::cout << "a[" << i << "] = " << resultView[i].a() << '\n';
  }

  return EXIT_SUCCESS;
}