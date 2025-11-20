// File: pimCmdFuse.cpp
// PIMeval Simulator - PIM API Fusion
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimCmdFuse.h"
#include "pimDevice.h"
#include <cstdio>


//! @brief  Pim CMD: PIM API Fusion
bool
pimCmdFuse::execute()
{
  if (m_debugCmds) {
    std::printf("PIM-Cmd: API Fusion\n");
  }

  // Functional simulation
  m_device->startCmdFuse(m_prog.m_apis.size());
  bool success = true;
  for (auto& api : m_prog.m_apis) {
    PimStatus status = api();
    if (status != PIM_OK) {
      success = false;
      break;
    }
  }

  m_device->clearFuseFlag();
  // Analyze API fusion opportunities
  success = success && updateStats();
  m_device->clearFusedCmds();
  return success;
}

//! @brief  Pim CMD: PIM API Fusion - update stats
bool
pimCmdFuse::updateStats() const
{
  bool success = true;
  for (auto& api : m_device->getFusedCmds()) {
    success &= api->updateStats();
  }
  return true;
}

