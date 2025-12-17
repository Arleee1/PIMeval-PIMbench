#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cassert>
#include "libpimeval.h"

typedef int PimObjGrid; // Could change definition later

//! @todo Maybe add enum to define how to allocate cores
//! @brief Define grid of cores, with specified sizes of data per core - allocates with locality awareness
//! @param allocType: type of allocation (e.g. PIM_ALLOC_AUTO)
//! @param dataType: type of data to be allocated
//! @param numCoresVertical: number of cores to allocate in the vertical direction
//! @param numCoresHorizontal: number of cores to allocate in the horizontal direction
//! @param numElementsPerCoreVertical: number of elements to allocate per core in the vertical direction
//! @param numElementsPerCoreHorizontal: number of elements to allocate per core in the horizontal direction
//! @return PimObjGrid representing the allocated grid of cores
PimObjGrid pimAllocGrid(PimAllocEnum allocType, PimDataType dataType, size_t numCoresVertical, size_t numCoresHorizontal,
                        size_t numElementsPerCoreVertical, size_t numElementsPerCoreHorizontal);

//! @brief Allocates more rows in each core of an associated PimObjGrid
//! @param assocGrid: PimObjGrid to associate with
//! @param dataType: type of data to be allocated
//! @param numElementsPerCoreVertical: number of additional elements to allocate per core in the vertical direction
//! @return PimObjGrid representing the newly allocated grid of cores associated with assocGrid
PimObjGrid pimAllocGridAssociated(PimObjGrid assocGrid, PimDataType dataType, size_t numElementsPerCoreVertical);

//! @param src: flattened 2D array
//! @param destGrid. The PimObjGrid allocated by pimAllocGrid/pimAllocGridAssociated
//! @param srcWidth: width of the source array in host memory
//! @param srcHeight: height of the source array in host memory
//! @param return: 1D vector of PimObjId - Vector of PimObjIds represents one tile, i.e. actions done to every tile
//!       e.g. calling pimAdd on the elements of the return vector will apply to every tile in the Grid
std::vector<PimObjId> pimCopyHostToGrid(const void* src, PimObjGrid destGrid, uint64_t srcWidth, uint64_t srcHeight);

//! @param srcGrid: PimObjGrid allocated by pimAllocGrid
//! @param dest: flattened 2D array in host memory
//! @param destWidth: width of the destination array in host memory
//! @param destHeight: height of the destination array in host memory
//! @return PimStatus indicating success or failure
PimStatus pimCopyGridToHost(PimObjGrid srcGrid, void* dest, uint64_t destWidth, uint64_t destHeight);

//! @param grid: PimObjGrid allocated by pimAllocGrid
//! @return PimStatus indicating success or failure
PimStatus pimFreeGrid(PimObjGrid grid);

void exampleUsage() {
    PimStatus status;

    const uint64_t srcWidth = 1024;
    const uint64_t srcHeight = 1024;
    const uint64_t tileWidth = 256;
    const uint64_t tileHeight = 256;

    const uint64_t numCoresVertical = srcHeight / tileHeight; // 4
    const uint64_t numCoresHorizontal = srcWidth / tileWidth; // 4

    PimObjGrid grid = pimAllocGrid(PIM_ALLOC_AUTO, PIM_INT32, numCoresVertical, numCoresHorizontal, tileHeight, tileWidth); // 4x4 grid of cores, each core with 256x256 elements
    
    int* src = (int*) std::malloc(srcWidth * srcHeight * sizeof(int)); // Example source data
    
    std::vector<PimObjId> pimObjs = pimCopyHostToGrid(src, grid, srcWidth, srcHeight);
    
    status = pimAdd(pimObjs[0], pimObjs[1], pimObjs[2]);
    assert(status == PIM_OK);

    int* dest = (int*) std::malloc(srcWidth * srcHeight * sizeof(int)); // Example destination data
    status = pimCopyGridToHost(grid, dest, srcWidth, srcHeight);
    assert(status == PIM_OK);

    // dest should now contain the result of the addition, applied to each tile in the grid

    status = pimFreeGrid(grid);
    assert(status == PIM_OK);

    std::free(src);
    std::free(dest);
}