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
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t iterations;
  uint64_t gridWidth;
  uint64_t gridHeight;
  uint64_t radius;
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./stencil.out [options]"
          "\n"
          "\n    -n    iterations (default=10 iterations)"
          "\n    -x    grid width (default=2048 elements)"
          "\n    -y    grid height (default=2048 elements)"
          "\n    -r    stencil radius (default=1)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a 2d array (default=random)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.iterations = 10;
  p.gridWidth = 2048;
  p.gridHeight = 2048;
  p.radius = 1;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:x:y:r:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.iterations = strtoull(optarg, NULL, 0);
      break;
    case 'x':
      p.gridWidth = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.gridHeight = strtoull(optarg, NULL, 0);
      break;
    case 'r':
      p.radius= strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
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

//! @brief  Sums the neighbors of each element in a stencil row to compute the horizontal stencil sum
//!
//! Sums radius number of elemements to the left and right of center element, including center element
//! Puts each result pimRowSum[i] where i is the center index
//! Formula: pimRowSum[i] = Σ (j ∈ [i-radius, i+radius]) mid[j]
//! Works by shifting mid to the left and right and adding shifted versions
//! @param[in]  mid  PIM row to be summed
//! @param[out]  pimRowSum  The resultant PIM object to place the sum into
//! @param[in,out]  shiftBackup  Temporary PIM object used for calculations
//! @param[in]  radius  The stencil radius
void sumStencilRow(PimObjId mid, PimObjId pimRowSum, PimObjId shiftBackup, const uint64_t radius) {
  PimStatus status;

  if(radius == 0) {
    return;
  }
  
  status = pimCopyObjectToObject(mid, shiftBackup);
  assert (status == PIM_OK);

  status = pimShiftElementsRight(shiftBackup, true);
  assert (status == PIM_OK);

  status = pimAdd(mid, shiftBackup, pimRowSum);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=1; shiftIter<radius; ++shiftIter) {
    status = pimShiftElementsRight(shiftBackup, true);
    assert (status == PIM_OK);

    status = pimAdd(pimRowSum, shiftBackup, pimRowSum);
    assert (status == PIM_OK);
  }

  status = pimCopyObjectToObject(mid, shiftBackup);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=0; shiftIter<radius; ++shiftIter) {
    status = pimShiftElementsLeft(shiftBackup, true);
    assert (status == PIM_OK);

    status = pimAdd(pimRowSum, shiftBackup, pimRowSum);
    assert (status == PIM_OK);
  }
}

struct VertChunkPim {
  std::vector<PimObjId> rowsInSumCircularQueue;
  std::vector<PimObjId> workingPimMemory;
  uint64_t firstRowIdxSrc;
  PimObjId tmpPim;
  PimObjId runningSum;

  VertChunkPim(uint64_t pimAllocWidth, uint64_t radius, uint64_t firstRowIdxSrc, uint64_t numRows)
      : firstRowIdxSrc(firstRowIdxSrc) {

    tmpPim = pimAlloc(PIM_ALLOC_AUTO, pimAllocWidth, PIM_FP32);
    assert(tmpPim != -1);
    runningSum = pimAllocAssociated(tmpPim, PIM_FP32);
    assert(runningSum != -1);

    rowsInSumCircularQueue.resize(2*radius+1);
    for(uint64_t i=0; i<rowsInSumCircularQueue.size(); ++i) {
      rowsInSumCircularQueue[i] = pimAllocAssociated(tmpPim, PIM_FP32);
      assert(rowsInSumCircularQueue[i] != -1);
    }

    workingPimMemory.resize(numRows);
    for(uint64_t i=0; i<workingPimMemory.size(); ++i) {
      workingPimMemory[i] = pimAllocAssociated(tmpPim, PIM_FP32);
      assert(workingPimMemory[i] != -1);
    }
  }

  void copyToPim(const std::vector<std::vector<float>> &srcHost) {
    for(uint64_t idx = 0; idx < workingPimMemory.size(); ++idx) {
      PimStatus status = pimCopyHostToDevice((void*) srcHost[firstRowIdxSrc + idx].data(), workingPimMemory[idx]);
      assert (status == PIM_OK);
    }
  }

  void copyFromPim(std::vector<std::vector<float>> &dstHost, const uint64_t numOverlap) {
    for(uint64_t idx = numOverlap; idx < workingPimMemory.size() - numOverlap; ++idx) {
      PimStatus status = pimCopyDeviceToHost(workingPimMemory[idx], (void*) dstHost[firstRowIdxSrc + idx].data());
      assert (status == PIM_OK);
    }
  }

  //! @brief  Computes one iteration of one chunk of the stencil
  //!
  //! Uses circular queue to compute window sums
  //! Adds the next row to the front of the queue and to the sum
  //! Takes the sum (divided by the stencil area) as the result from the row
  //! Subtracts the back of the queue from the sum
  //! Pops from the queue back of the queue
  //! Repeats until done
  //! @param[in]  workingPimMemory  PIM rows in the stencil chunk
  //! @param[in]  rowsInSumCircularQueue  Queue used for keeping track of running sum of rows vertically
  //! @param[in,out]  tmpPim  Temporary PIM object used for calculations
  //! @param[in,out]  runningSum Temporary PIM object used for keeping track of the current running (vertical) sum
  //! @param[in]  stencilAreaToMultiplyPim This algorithm computes stencil average, thus each element in the result must be divided by the stencil area. This is done by multiplying by the inverse.
  //! @param[in]  radius  The stencil radius
  void computeStencilIteration(const uint64_t stencilAreaToMultiplyPim, const uint64_t radius) {
    PimStatus status;

    uint64_t circularQueueBot = 0;
    uint64_t circularQueueTop = 0;

    sumStencilRow(workingPimMemory[0], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
    ++circularQueueTop;
    sumStencilRow(workingPimMemory[1], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
    ++circularQueueTop;
    status = pimAdd(rowsInSumCircularQueue[0], rowsInSumCircularQueue[1], runningSum);
    assert (status == PIM_OK);

    // At this point:
    // circularQueueBot = 0
    // circularQueueTop = 2
    // rowsInSumCircularQueue[0] = workingPimMemory[0] horizontally summed
    // rowsInSumCircularQueue[1] = workingPimMemory[1] horizontally summed
    // runningSum = sum of first two rows horizontally summed

    for(uint64_t i=2; i<2*radius; ++i) {
      sumStencilRow(workingPimMemory[i], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
      status = pimAdd(runningSum, rowsInSumCircularQueue[circularQueueTop], runningSum);
      assert (status == PIM_OK);
      ++circularQueueTop;
    }

    // At this point:
    // circularQueueBot = 0
    // circularQueueTop = 2*radius
    // rowsInSumCircularQueue[0...2*radius] are occupied with workingPimMemory[0...2*radius] horizontally summed
    // runningSum = sum of rows [0...2*radius] horizontally summed

    uint64_t nextRowToAdd = 2*radius; // The index of the next row to add to the queue and to the running sum

    // Loops over the rest of the rows in the current chunk, vertically
    // Each iteration, finds horizontal sum of the next row (nextRowToAdd)
    // Places this horizontal sum at the front of the queue (at position circularQueueTop)
    // Adds the horizontal sum to the runningSum
    // Places runningSum/stencilArea into the workingPimMemory as the final result for the row
    // If neccessary, subtracts the row from the back of the queue from the runningSum

    for(uint64_t row=radius; row<workingPimMemory.size()-radius; ++row) {
      sumStencilRow(workingPimMemory[nextRowToAdd], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);

      status = pimAdd(runningSum, rowsInSumCircularQueue[circularQueueTop], runningSum);
      assert (status == PIM_OK);

      circularQueueTop = (1+circularQueueTop) % rowsInSumCircularQueue.size();
      ++nextRowToAdd;

      status = pimMulScalar(runningSum, workingPimMemory[row], stencilAreaToMultiplyPim);
      assert (status == PIM_OK);
      
      if(row+1<workingPimMemory.size()-radius) {
        status = pimSub(runningSum, rowsInSumCircularQueue[circularQueueBot], runningSum);
        assert (status == PIM_OK);
        circularQueueBot = (1+circularQueueBot) % rowsInSumCircularQueue.size();
      }
    }
  }
};

void pimMove(std::vector<float>& hostTmpRow, PimObjId pimSrc, PimObjId pimDst) {
  PimStatus status = pimCopyDeviceToHost(pimSrc, hostTmpRow.data());
  assert(status == PIM_OK);
  status = pimCopyHostToDevice(hostTmpRow.data(), pimDst);
  assert(status == PIM_OK);
}

//! @brief  Computes a stencil pattern over a 2d array
//! @param[in]  srcHost  The input stencil grid
//! @param[out]  dstHost  The resultant stencil grid
//! @param[in]  numAssociable  Number of float 32 PIM objects that can be associated with each other
//! @param[in]  numElementsHorizontal  Number of float 32 PIM objects that can be placed in a PIM row without creating shifting issues
//! @param[in]  iterations  Number of iterations to run the stencil pattern for
//! @param[in]  radius  The radius of the stencil pattern
void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost, const uint64_t numAssociable,
              const uint64_t numElementsHorizontal, const uint64_t iterations, const uint64_t radius) {
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());

  std::vector<std::vector<float>> tmpGrid;
  tmpGrid.resize(srcHost.size(), std::vector<float>(srcHost[0].size()));

  const uint64_t gridWidth = srcHost[0].size();

  const uint64_t stencilAreaInt = (2 * radius + 1) * (2 * radius + 1);
  const float stencilAreaFloat = 1.0f / static_cast<float>(stencilAreaInt);
  uint32_t tmp;
  std::memcpy(&tmp, &stencilAreaFloat, sizeof(float));
  const uint64_t stencilAreaToMultiplyPim = static_cast<uint64_t>(tmp);
  const uint64_t pimAllocWidth = gridWidth;

  const uint64_t maxRowsPerVertChunk = numAssociable - (2*radius + 1) - 2;
  const uint64_t numOverlap = radius;
  uint64_t numVertChunks;
  if (srcHost.size() <= maxRowsPerVertChunk) {
    numVertChunks = 1;
  } else if (srcHost.size() <= 2*(maxRowsPerVertChunk - numOverlap)) {
    numVertChunks = 2;
  } else {
    const uint64_t firstAndLastChunkRows = 2 * (maxRowsPerVertChunk - numOverlap);
    const uint64_t remainingRows = srcHost.size() - firstAndLastChunkRows;
    const uint64_t middleChunkSize = maxRowsPerVertChunk - 2*numOverlap;
    const uint64_t numMiddleChunks = (remainingRows + middleChunkSize - 1) / middleChunkSize;
    numVertChunks = 2 + numMiddleChunks;
  }

  std::vector<VertChunkPim> vertChunks;
  vertChunks.reserve(numVertChunks);

  for(uint64_t chunkIdx=0; chunkIdx<numVertChunks; ++chunkIdx) {
    const uint64_t firstRowIdxSrc = chunkIdx*(maxRowsPerVertChunk - 2*numOverlap);
    const uint64_t lastRowIdxSrc = std::min(srcHost.size(), firstRowIdxSrc + maxRowsPerVertChunk);
    const uint64_t numRows = lastRowIdxSrc - firstRowIdxSrc;
    vertChunks.emplace_back(pimAllocWidth, radius, firstRowIdxSrc, numRows);
    vertChunks.back().copyToPim(srcHost);
  }

  std::vector<float> hostTmpRow(gridWidth, 0.0f);

  for(uint64_t iter=0; iter<iterations; ++iter) {
    for(auto& vertChunk : vertChunks) {
      vertChunk.computeStencilIteration(stencilAreaToMultiplyPim, radius);
    }
    if(iter+1<iterations) {
      for(uint64_t chunkIdx=0; chunkIdx<numVertChunks-1; ++chunkIdx) {
        std::vector<PimObjId>& above = vertChunks[chunkIdx].workingPimMemory;
        std::vector<PimObjId>& below = vertChunks[chunkIdx+1].workingPimMemory;

        for(uint64_t row=0; row<numOverlap; ++row) {
          PimObjId pimSrc;
          PimObjId pimDst;
          
          pimSrc = above[above.size() - 2 * numOverlap + row];
          pimDst = below[row];
          pimMove(hostTmpRow, pimSrc, pimDst);

          pimSrc = below[numOverlap + row];
          pimDst = above[above.size() - numOverlap + row];
          pimMove(hostTmpRow, pimSrc, pimDst);
        }
      }
    }
  }

  for(auto& vertChunk : vertChunks) {
    vertChunk.copyFromPim(dstHost, numOverlap);
  }
}

void stencilCpu(std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst, const uint64_t iterations, const uint64_t radius) {
  const uint64_t stencilAreaInt = (2 * radius + 1) * (2 * radius + 1);
  const float stencilAreaInverseFloat = 1.0f / static_cast<float>(stencilAreaInt);

  for(uint64_t iter=1; iter<=iterations; ++iter) {
    // Only compute when stencil is fully in range
    const uint64_t startY = radius*iter;
    const uint64_t endY = src.size() - startY;
    const uint64_t startX = radius*iter;
    const uint64_t endX = src[0].size() - startX;
    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=gridY-radius; stencilY<=gridY+radius; ++stencilY) {
          for(uint64_t stencilX=gridX-radius; stencilX<=gridX+radius; ++stencilX) {
            resCPU += src[stencilY][stencilX];
          }
        }
        dst[gridY][gridX] = resCPU * stencilAreaInverseFloat;
      }
    }
    std::swap(src, dst);
  }
  std::swap(src, dst);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::cout << "Running PIM stencil for grid: " << params.gridHeight << "x" << params.gridWidth << std::endl;
  std::cout << "Stencil Radius: " << params.radius << ", Number of Iterations: " << params.iterations << std::endl;
  if constexpr(true) {
    std::cout << "Stencil does not use cross region communication" << std::endl;
  } else {
    std::cout << "Stencil uses cross region communication" << std::endl;
  }

  std::vector<std::vector<float>> x, y;

  if (params.inputFile == nullptr)
  {
    // Fill in random grid
    x.resize(params.gridHeight, std::vector<float>(params.gridWidth));

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 10000.0f);
      
      #pragma omp for
      for(size_t i=0; i<params.gridHeight; ++i) {
        for(size_t j=0; j<params.gridWidth; ++j) {
          x[i][j] = dist(gen);
        }
      }
    }
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  y.resize(x.size(), std::vector<float>(x[0].size()));

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  constexpr uint64_t bitsPerElement = 32;

  uint64_t numAssociable = 2 * deviceProp.numRowPerSubarray;
  if(!deviceProp.isHLayoutDevice) {
    numAssociable /= bitsPerElement;
  }

  uint64_t numElementsHorizontal;
  if(deviceProp.isHLayoutDevice) {
    switch(deviceProp.simTarget) {
      case PIM_DEVICE_FULCRUM:
      case PIM_DEVICE_BANK_LEVEL:
        numElementsHorizontal = deviceProp.numColPerSubarray / bitsPerElement;
        break;
      default:
        std::cerr << "Stencil unimplemented for simulation target: " << deviceProp.simTarget << std::endl;
        std::exit(1);
    }
  } else {
    numElementsHorizontal = deviceProp.numColPerSubarray;
  }

  stencil(x, y, numAssociable, numElementsHorizontal, params.iterations, params.radius);

  if (params.shouldVerify) 
  {
    std::vector<std::vector<float>> cpuY(y.size(), std::vector<float>(y[0].size()));
    stencilCpu(x, cpuY, params.iterations, params.radius);

    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = params.radius * params.iterations;
    const uint64_t endY = params.gridHeight - startY;
    const uint64_t startX = params.radius * params.iterations;
    const uint64_t endX = params.gridWidth - startX;

    std::cout << std::fixed << std::setprecision(10);

    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        constexpr float acceptableDelta = 0.1f;
        if (std::abs(cpuY[gridY][gridX] - y[gridY][gridX]) > acceptableDelta)
        {
          #pragma omp critical
          {
            std::cout << "Wrong answer: " << y[gridY][gridX] << " (expected " << cpuY[gridY][gridX] << ") at position (" << gridX << ", " << gridY << ")" << std::endl;
            ok = false;
          }
        }
      }
    }
    if(ok) {
      std::cout << "Correct for stencil!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}