#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cassert>
#include "libpimeval.h"

//! @brief Represents a grid of PimObjIds, possibly spread across multiple PIM cores
typedef std::vector<PimObjId> PimObjGrid;

//! @todo define and implement allocation strategies
//! @brief Different strategies for allocating PimObjIds in a PimObjGrid - determines layout of cores in memory
enum PimAllocationStrategy {
  PIM_ALLOCATION_STRATEGY_1 = 0,
  PIM_ALLOCATION_STRATEGY_2
};

//! @brief Define grid of cores, with specified sizes of data per core - allocates with locality awareness
//! @param allocType: type of allocation (e.g. PIM_ALLOC_AUTO)
//! @param dataType: type of data to be allocated
//! @param numCoresVertical: number of cores to allocate in the vertical direction
//! @param numCoresHorizontal: number of cores to allocate in the horizontal direction
//! @param numElementsPerCoreVertical: number of elements to allocate per core in the vertical direction
//! @param numElementsPerCoreHorizontal: number of elements to allocate per core in the horizontal direction
//! @param allocationStrategy: strategy to use for allocation, determines the layout of cores within memory
//! @return PimObjGrid representing the allocated grid of cores
PimObjGrid pimAllocGrid(PimAllocEnum allocType, PimDataType dataType, size_t numCoresVertical,
                                    size_t numCoresHorizontal, size_t numElementsPerCoreVertical,
                                    size_t numElementsPerCoreHorizontal,
                                    PimAllocationStrategy allocationStrategy = PIM_ALLOCATION_STRATEGY_1);

//! @brief Allocates more PimObjs associated with an existing PimObjId
//! @param assocId: PimObjId to associate with
//! @param dataType: type of data to be allocated
//! @param numObjs: number of additional PimObjs to allocate
//! @return PimObjGrid of newly allocated PimObjIds
PimObjGrid pimAllocGridAssociated(PimObjId assocId, PimDataType dataType, size_t numObjs);

//! @brief Copies data from a flattened 2D array in host memory to a PimObjGrid
//! @param src: flattened 2D array
//! @param destGrid: The PimObjGrid allocated by pimAllocGrid/pimAllocGridAssociated
//! @param idxBeginX: starting index in the X direction (columns) to copy, per core
//! @param idxEndX: ending index in the X direction (columns) to copy, per core
//! @param idxBeginY: starting index in the Y direction (rows) to copy, per core
//! @param idxEndY: ending index in the Y direction (rows) to copy, per core
//! @return PimStatus indicating success or failure
PimStatus pimCopyHostToGrid(const void* src, PimObjGrid& destGrid, uint64_t idxBeginX = 0, uint64_t idxEndX = 0,
                                       uint64_t idxBeginY = 0, uint64_t idxEndY = 0);

//! @brief Copies data from a PimObjGrid to a flattened 2D array in host memory
//! @param srcGrid: PimObjGrid allocated by pimAllocGrid
//! @param dest: flattened 2D array in host memory
//! @param idxBeginX: starting index in the X direction (columns) to copy, per core
//! @param idxEndX: ending index in the X direction (columns) to copy, per core
//! @param idxBeginY: starting index in the Y direction (rows) to copy, per core
//! @param idxEndY: ending index in the Y direction (rows) to copy, per core
//! @return PimStatus indicating success or failure
PimStatus pimCopyGridToHost(PimObjGrid srcGrid, void* dest, uint64_t idxBeginX = 0, uint64_t idxEndX = 0,
                           uint64_t idxBeginY = 0, uint64_t idxEndY = 0);

//! @brief Frees the PimObjGrid and all associated PimObjIds
//! @param grid: PimObjGrid allocated by pimAllocGrid
//! @return PimStatus indicating success or failure
PimStatus pimFreeGrid(PimObjGrid grid);




// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Example usage of the above APIs
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void copyHaloToAdjacent(PimObjGrid& grid) {
    // Not yet implemented: would copy halo regions between adjacent cores in the grid
}

void stencilIteration(PimObjGrid& grid) {
    // Not yet implemented: would perform stencil computation on each tile in the grid (possibly in parallel)
    // Could mostly use existing stencil code
}

void gameOfLifeIteration(PimObjGrid& grid) {
    // Not yet implemented: would perform Game of Life computation on each tile in the grid (possibly in parallel)
    // Could mostly use existing Game of Life code
}

void exampleStencilUsage() {
    PimStatus status;

    const uint64_t srcWidth = 1024;
    const uint64_t srcHeight = 1024;
    const uint64_t tileWidth = 256;
    const uint64_t tileHeight = 256;
    const uint64_t numIterations = 10;

    const uint64_t numCoresVertical = srcHeight / tileHeight; // 4
    const uint64_t numCoresHorizontal = srcWidth / tileWidth; // 4

    // 4x4 grid of cores, each core with 256x256 elements plus 1-element halo on each side
    PimObjGrid grid = pimAllocGrid(PIM_ALLOC_AUTO, PIM_FP32, numCoresVertical, numCoresHorizontal, tileHeight + 2, tileWidth + 2);
    assert(grid.size() > 0);

    float* src = (float*) std::malloc(srcWidth * srcHeight * sizeof(float)); // Example source data

    // Copy to PIM, skipping halo regions
    PimStatus status = pimCopyHostToGrid(src, grid, 1, tileWidth + 1, 1, tileHeight + 1);
    // Fill the halo regions
    copyHaloToAdjacent(grid);

    for(size_t iter = 0; iter < numIterations; ++iter) {
        stencilIteration(grid);
        copyHaloToAdjacent(grid);
    }

    float* dest = (float*) std::malloc(srcWidth * srcHeight * sizeof(float)); // Example destination data

    // Only copy back the non-halo region
    status = pimCopyGridToHost(grid, dest, 1, tileWidth + 1, 1, tileHeight + 1);
    assert(status == PIM_OK);

    // dest should now have the results of the stencil computation

    status = pimFreeGrid(grid);
    assert(status == PIM_OK);

    std::free(src);
    std::free(dest);
}

void exampleGameOfLifeUsage() {
    PimStatus status;

    const uint64_t srcWidth = 1024;
    const uint64_t srcHeight = 1024;
    const uint64_t tileWidth = 256;
    const uint64_t tileHeight = 256;
    const uint64_t numIterations = 10;

    const uint64_t numCoresVertical = srcHeight / tileHeight; // 4
    const uint64_t numCoresHorizontal = srcWidth / tileWidth; // 4

    // Allocate 1-element halo on each side
    // Use PIM_BOOL for Game of Life
    PimObjGrid grid = pimAllocGrid(PIM_ALLOC_AUTO, PIM_BOOL, numCoresVertical, numCoresHorizontal, tileHeight + 2, tileWidth + 2);
    assert(grid.size() > 0);

    // Stores sums during GOL computation
    PimObjGrid sumAccumulator = pimAllocGridAssociated(grid[0], PIM_UINT8, 1);
    assert(sumAccumulator.size() > 0);

    float* src = (float*) std::malloc(srcWidth * srcHeight * sizeof(float)); // Example source data

    // Copy to PIM, skipping halo regions
    PimStatus status = pimCopyHostToGrid(src, grid, 1, tileWidth + 1, 1, tileHeight + 1);

    // Fill the halo regions
    copyHaloToAdjacent(grid);

    for(size_t iter = 0; iter < numIterations; ++iter) {
        gameOfLifeIteration(grid);
        copyHaloToAdjacent(grid);
    }

    float* dest = (float*) std::malloc(srcWidth * srcHeight * sizeof(float)); // Example destination data

    // Only copy back the non-halo region
    status = pimCopyGridToHost(grid, dest, 1, tileWidth + 1, 1, tileHeight + 1);
    assert(status == PIM_OK);

    status = pimFreeGrid(grid);
    assert(status == PIM_OK);

    std::free(src);
    std::free(dest);
}