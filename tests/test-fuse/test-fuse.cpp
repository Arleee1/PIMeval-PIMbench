// Test: Fuse
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <cassert>

bool testFuse(PimDeviceEnum deviceType, bool shouldFuse)
{
  unsigned numRanks = 2;
  unsigned numBankPerRank = 2;
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  PimStatus status = pimCreateDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  unsigned numElements = 1000;

  std::vector<int> src1A(numElements);
  std::vector<int> src1B(numElements);
  std::vector<int> dest1(numElements);
  PimObjId dest1pim = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  assert(dest1pim != -1);
  PimObjId src1Apim = pimAllocAssociated(dest1pim, PIM_INT32);
  assert(src1Apim != -1);
  PimObjId src1Bpim = pimAllocAssociated(dest1pim, PIM_INT32);
  assert(src1Bpim != -1);
  

  std::vector<int> src2A(numElements);
  std::vector<int> src2B(numElements);
  std::vector<int> dest2(numElements);
  PimObjId dest2pim = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
  assert(dest2pim != -1);
  PimObjId src2Apim = pimAllocAssociated(dest2pim, PIM_INT32);
  assert(src2Apim != -1);
  PimObjId src2Bpim = pimAllocAssociated(dest2pim, PIM_INT32);
  assert(src2Bpim != -1);

  // assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1A[i] = i;
    src1B[i] = i * 3 + 7;
    src2A[i] = i * 2 - 10;
    src2B[i] = i * 4 + 5;
  }

  status = pimCopyHostToDevice((void*)src1A.data(), src1Apim);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src1B.data(), src1Bpim);
  assert(status == PIM_OK);

  status = pimCopyHostToDevice((void*)src2A.data(), src2Apim);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2B.data(), src2Bpim);
  assert(status == PIM_OK);

  if(shouldFuse) {
    PimProg prog;
    prog.add(pimAdd, src1Apim, src1Bpim, dest1pim);
    prog.add(pimAdd, src2Apim, src2Bpim, dest2pim);
    status = pimFuse(prog);
    assert(status == PIM_OK);
  } else {
    status = pimAdd(src1Apim, src1Bpim, dest1pim);
    assert(status == PIM_OK);
    status = pimAdd(src2Apim, src2Bpim, dest2pim);
    assert(status == PIM_OK);
  }


  status = pimCopyDeviceToHost(dest1pim, (void*)dest1.data());
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(dest2pim, (void*)dest2.data());
  assert(status == PIM_OK);

  for (unsigned i = 0; i < numElements; ++i) {
    if (dest1[i] != src1A[i] + src1B[i]) {
      return false;
    }
    if (dest2[i] != src2A[i] + src2B[i]) {
      return false;
    }
  }

  pimShowStats();
  pimResetStats();
  pimDeleteDevice();
  return true;
}

int main()
{
  std::cout << "INFO: Fuse Test" << std::endl;

  bool ok = true;
  bool shouldFuse = false;
  ok &= testFuse(PIM_DEVICE_BITSIMD_V, shouldFuse);
  std::cout << "Suceeded for BITSiMD-V, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_BITSIMD_V_AP, shouldFuse);
  std::cout << "Suceeded for BITSiMD-V-AP, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_BANK_LEVEL, shouldFuse);
  std::cout << "Suceeded for Bank-Level PIM, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_FULCRUM, shouldFuse);
  std::cout << "Suceeded for Fulcrum PIM, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  shouldFuse = true;
  ok &= testFuse(PIM_DEVICE_BITSIMD_V, shouldFuse);
  std::cout << "Suceeded for BITSiMD-V, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_BITSIMD_V_AP, shouldFuse);
  std::cout << "Suceeded for BITSiMD-V-AP, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_BANK_LEVEL, shouldFuse);
  std::cout << "Suceeded for Bank-Level PIM, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  ok &= testFuse(PIM_DEVICE_FULCRUM, shouldFuse);
  std::cout << "Suceeded for Fulcrum PIM, fuse: " << (shouldFuse ? "ON" : "OFF") << std::endl;
  std::cout << (ok ? "ALL PASSED!" : "FAILED!") << std::endl;

  return 0;
}