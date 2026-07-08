#include <cstdlib>
#include <memory>
#include <tuple>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(InputLayoutTemplate,
  SOA_COLUMN(float, x),
  SOA_COLUMN(float, y),
  SOA_COLUMN(float, z))

GENERATE_SOA_LAYOUT(ResultLayoutTemplate,
  SOA_COLUMN(float, a))

using InputLayout = InputLayoutTemplate<>;
using ResultLayout = ResultLayoutTemplate<>;

using ViewInput = InputLayout::View;
using ViewResult = ResultLayout::View;

int main(){

  constexpr int slSize = 128;

  const std::size_t inputBufferSize = InputLayout::computeDataSize(slSize);
  const std::size_t resultBufferSize = ResultLayout::computeDataSize(slSize);

  std::unique_ptr<std::byte, decltype(std::free) *> inputBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(InputLayout::alignment, inputBufferSize)), std::free};
  std::unique_ptr<std::byte, decltype(std::free) *> resultBuffer{
      reinterpret_cast<std::byte *>(aligned_alloc(ResultLayout::alignment, resultBufferSize)), std::free};

  InputLayout input{inputBuffer.get(), slSize};
  ResultLayout result{resultBuffer.get(), slSize};

  ViewInput inputView{input};
  ViewResult resultView{result};

  for (int i = 0; i < slSize; ++i) {
    inputView[i] = {static_cast<float>(i), static_cast<float>(i) * 2.f, static_cast<float>(i) * 3.f};
  }

  for (int i = 0; i < slSize; ++i) {
    resultView[i].a() = inputView[i].y() * inputView[i].x() + inputView[i].z();
  }

  return EXIT_SUCCESS;
}