// Test: Alloc Location
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <cassert>

bool testAllocLocation(PimDeviceEnum deviceType)
{
  unsigned numRanks = 4;
  unsigned numBankPerRank = 4;
  unsigned numSubarrayPerBank = 16;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  PimDeviceProperties deviceProp;
  status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  constexpr uint64_t numElements = 10;

  PimAllocLocation loc = {2, -1, -1};
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT8, loc);
  PimObjId obj2 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT8, loc);

  PimAllocLocation allocLocation1;
  PimAllocLocation allocLocation2;

  status = pimGetObjLocation(obj1, &allocLocation1);
  assert(status == PIM_OK);
  status = pimGetObjLocation(obj2, &allocLocation2);
  assert(status == PIM_OK);
  assert(allocLocation1.rank == 2 && allocLocation2.rank == 2);

  loc = {1, 1, -1};
  obj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT8, loc);
  obj2 = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_UINT8, loc);

  status = pimGetObjLocation(obj1, &allocLocation1);
  assert(status == PIM_OK);
  status = pimGetObjLocation(obj2, &allocLocation2);
  assert(status == PIM_OK);
  assert(allocLocation1.rank == 1 && allocLocation2.rank == 1);
  assert(allocLocation1.bank == allocLocation2.bank); // Should be same bank, however doesn't need to be original bank, if multiple banks per core

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return true;
}

int main()
{
  std::cout << "INFO: Alloc Location Test" << std::endl;

  bool ok = true;
  ok &= testAllocLocation(PIM_DEVICE_BITSIMD_V);
  std::cout << "Suceeded for BITSiMD-V" << std::endl;
  ok &= testAllocLocation(PIM_DEVICE_BITSIMD_V_AP);
  std::cout << "Suceeded for BITSiMD-V-AP" << std::endl;
  ok &= testAllocLocation(PIM_DEVICE_BANK_LEVEL);
  std::cout << "Suceeded for Bank-Level PIM" << std::endl;
  ok &= testAllocLocation(PIM_DEVICE_FULCRUM);
  std::cout << "Suceeded for Fulcrum PIM" << std::endl;
  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}