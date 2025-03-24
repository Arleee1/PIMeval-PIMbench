// Test: C++ version of the stencil
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#include <type_traits>
#include <queue>
#include <random>
#include <limits>
#include <algorithm>
#include <list>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t gridWidth;
  uint64_t gridHeight;
  uint64_t stencilWidth;
  uint64_t stencilHeight;
  uint64_t numLeft;
  uint64_t numAbove;
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./stencil.out [options]"
          "\n"
          "\n    -x    grid width (default=2048 elements)"
          "\n    -y    grid height (default=2048 elements)"
          "\n    -w    horizontal stencil size (default=3)"
          "\n    -d    vertical stencil size (default=3)"
          "\n    -l    number of elements to the left of the output element for the stencil pattern, must be less than the horizontal stencil size (default=1)"
          "\n    -a    number of elements above the output element for the stencil pattern, must be less than the vertical stencil size (default=1)"
          "\n    -i    input file containing a 2d array (default=random)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.gridWidth = 2048;
  p.gridHeight = 2048;
  p.stencilWidth = 3;
  p.stencilHeight = 3;
  p.numLeft = 1;
  p.numAbove = 1;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:w:d:l:a:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.gridWidth = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.gridHeight = strtoull(optarg, NULL, 0);
      break;
    case 'w':
      p.stencilWidth = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.stencilHeight = strtoull(optarg, NULL, 0);
      break;
    case 'l':
      p.numLeft = strtoull(optarg, NULL, 0);
      break;
    case 'a':
      p.numAbove = strtoull(optarg, NULL, 0);
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't');
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

inline __device__ uint64_t getIdxFromPos(const uint64_t x, const uint64_t y, const uint64_t width) {
  return y*width + x;
}

//! @brief  Computes a stencil average over a 2d array using CUDA. Grid cells who's stencil pattern lies partly outside of the input data range are undefined.
//! @param[in]  src  Device pointer to stencil input data
//! @param[out]  dst  Device pointer to stencil output data
//! @param[in]  toDivideBy  The number to divide the stencil sums by (Will be the number of cells in the stencil pattern for an average)
//! @param[in]  gridWidth  The width of the input and output grid
//! @param[in]  gridHeight  The height of the input and output grid
//! @param[in]  stencilWidth  The horizontal width of the stencil average rectangle
//! @param[in]  stencilHeight  The vertical height of the stencil average rectangle
//! @param[in]  numLeft  The number of elements to the left of the result grid element
//! @param[in]  numAbove  The number of elements to the right of the result grid element
template <typename StencilType>
__global__ void rectangleStencilAverage(
  const StencilType* src,
  StencilType* dst,
  const StencilType toDivideBy,
  const uint64_t gridWidth,
  const uint64_t gridHeight,
  const uint64_t stencilWidth,
  const uint64_t stencilHeight,
  const uint64_t numLeft,
  const uint64_t numRight,
  const uint64_t numAbove,
  const uint64_t numBelow
) {
  const uint64_t xPos = blockDim.x * blockIdx.x + threadIdx.x + numLeft;
	const uint64_t yPos = blockDim.y * blockIdx.y + threadIdx.y + numAbove;

  if((xPos + numRight) >= gridWidth || (yPos + numBelow) >= gridHeight) {
    return;
  }

  const uint64_t idx = getIdxFromPos(xPos, yPos, gridWidth);

  StencilType output = 0;
  for(uint64_t y=yPos-numAbove; y<=yPos+numBelow; ++y) {
    for(uint64_t x=xPos-numLeft; x<=xPos+numRight; ++x) {
      output += src[getIdxFromPos(x, y, gridWidth)];
    }
  }

  dst[idx] = output / toDivideBy;
}

//! @brief  Computes a stencil pattern over a 2d array
//! @param[in]  srcHost  The input stencil grid
//! @param[in]  dstHost  The resultant stencil grid
//! @param[in]  gridWidth  The width of the stencil grid
//! @param[in]  gridHeight  The height of the stencil grid
//! @param[in]  stencilWidth  The width of the stencil average rectangle
//! @param[in]  stencilHeight  The height of the stencil average rectangle
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  numAbove  The number of elements above the output element in the stencil pattern
void stencil(
  const std::vector<float> &srcHost,
  std::vector<float> &dstHost,
  const uint64_t gridWidth,
  const uint64_t gridHeight,
  const uint64_t stencilWidth,
  const uint64_t stencilHeight,
  const uint64_t numLeft,
  const uint64_t numAbove
) {

  assert(!srcHost.empty());
  assert(srcHost.size() == dstHost.size());
  assert(gridWidth * gridHeight == srcHost.size());
  assert(numLeft < stencilWidth);
  assert(numAbove < stencilHeight);

  const uint64_t numBelow = stencilHeight - numAbove - 1;
  const uint64_t numRight = stencilWidth - numLeft - 1;
  const float toDivideBy = static_cast<float>(stencilWidth * stencilHeight);

  cudaError_t errorCode;

  float* srcGPU;
	float* dstGPU;
  const size_t gridSz = gridHeight * gridWidth * sizeof(float);
  errorCode = cudaMalloc((void **)&srcGPU, gridSz);
  if(errorCode != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << std::endl;
    std::exit(1);
  }
	errorCode = cudaMalloc((void **)&dstGPU, gridSz);
  if(errorCode != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << std::endl;
    std::exit(1);
  }

  errorCode = cudaMemcpy(srcGPU, srcHost.data(), gridSz, cudaMemcpyHostToDevice);
  if(errorCode != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << std::endl;
    std::exit(1);
  }

  dim3 dimBlock(32, 32);
  // Only compute grid cells where the stencil pattern is fully in range
  dim3 dimGrid((gridWidth - stencilWidth + dimBlock.x) / dimBlock.x, (gridHeight - stencilHeight + dimBlock.y) / dimBlock.y);
  
  rectangleStencilAverage<<<dimGrid, dimBlock>>>(
    srcGPU,
    dstGPU,
    toDivideBy,
    gridWidth,
    gridHeight,
    stencilWidth,
    stencilHeight,
    numLeft,
    numRight,
    numAbove,
    numBelow
  );

  errorCode = cudaGetLastError();
  if(errorCode != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << std::endl;
    std::exit(1);
  }

  errorCode = cudaMemcpy(dstHost.data(), dstGPU, gridSz, cudaMemcpyDeviceToHost);
  if(errorCode != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << std::endl;
    std::exit(1);
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::cout << "Running GPU stencil for grid: " << params.gridHeight << "x" << params.gridWidth << std::endl;
  std::cout << "Stencil Size: " << params.stencilHeight << "x" << params.stencilWidth << std::endl;
  std::cout << "Num Above: " << params.numAbove << ", Num Left: " << params.numLeft << std::endl;

  std::vector<float> x, y;
  if (params.inputFile == nullptr)
  {
    // Fill in random grid
    x.resize(params.gridHeight * params.gridWidth);

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 10000.0f);

      #pragma omp for
      for(size_t i=0; i<x.size(); ++i) {
        x[i] = dist(gen);
      }
    }
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  y.resize(x.size());

  stencil(
    x,
    y,
    params.gridWidth,
    params.gridHeight,
    params.stencilWidth,
    params.stencilHeight,
    params.numLeft,
    params.numAbove
  );

  if (params.shouldVerify)
  {
    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = params.numAbove;
    const uint64_t endY = params.gridHeight - (params.stencilHeight - params.numAbove - 1);
    const uint64_t startX = params.numLeft;
    const uint64_t endX = params.gridWidth - (params.stencilWidth - params.numLeft - 1);
    const uint64_t numBelow = params.stencilHeight - params.numAbove - 1;
    const uint64_t numRight = params.stencilWidth - params.numLeft - 1;
    const float toDivideBy = static_cast<float>(params.stencilWidth * params.stencilHeight);

    // CPU and GPU results are not exactly the same
    // TODO: Check if this is okay
    constexpr float acceptableDifference = 0.1f;

    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=gridY-params.numAbove; stencilY<=gridY+numBelow; ++stencilY) {
          for(uint64_t stencilX=gridX-params.numLeft; stencilX<=gridX+numRight; ++stencilX) {
            resCPU += x[stencilY * params.gridWidth + stencilX];
          }
        }
        resCPU /= toDivideBy;
        if (std::abs(resCPU - y[gridY * params.gridWidth + gridX]) > acceptableDifference)
        {
          #pragma omp critical
          {
            std::cout << std::fixed << std::setprecision(3) << "Wrong answer: " << y[gridY * params.gridWidth + gridX] << " (expected " << resCPU << ") at position " << gridX << ", " << gridY << std::endl;
            ok = false;
          }
        }
      }
    }
    if(ok) {
      std::cout << "Correct for stencil!" << std::endl;
    }
  }

  return 0;
}