#ifndef NEW_PIM_OPS_H
#define NEW_PIM_OPS_H

#include <cassert>

#include "libpimeval.h"

// Stores equality result in 1 bit
// Outputs to dest, only updates the dest_offset'th bit
// pim_scratch used internally, should not be needed in final implementation
PimStatus pimEQScalar1BitResult(PimObjId pim_scratch, PimObjId src, PimObjId dest, uint64_t dest_offset, uint64_t scalarValue) {

    PimStatus status;
    status = pimEQScalar(src, pim_scratch, scalarValue);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpReadRowToSa(pim_scratch, 0);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpWriteSaToRow(dest, dest_offset);
    return status;
}

// Copies 1 bit object to other 1 bit object (associated)
// src, src_offset should be replaced with a 1bit pim object
// dest, dest_offset should be replaced with a 1bit pim object
PimStatus pimCopyDeviceToDevice1Bit(PimObjId src, PimObjId src_offset, PimObjId dest, uint64_t dest_offset) {

    PimStatus status;
    status = pimOpReadRowToSa(src, src_offset);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpWriteSaToRow(dest, dest_offset);
    return status;
}

// Ors 2 1bit objects into a 3rd 1bit object (associated)
// All PimObjId parameters and offsets should be replaced with 1bit pim objects
// PimStatus pimOr1bit(PimObjId src1, uint64_t src1_offset, PimObjId src2, uint64_t src2_offset, PimObjId dest, uint64_t dest_offset) {

//     PimStatus status;
//     status = pimOpReadRowToSa(src1, src1_offset);
//     if(status != PIM_OK) {
//         return status;
//     }

//     status = pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
//     if(status != PIM_OK) {
//         return status;
//     }

//     status = pimOpReadRowToSa(src2, src2_offset);
//     if(status != PIM_OK) {
//         return status;
//     }

//     status = pimOpOr(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
//     if(status != PIM_OK) {
//         return status;
//     }

//     status = pimOpWriteSaToRow(dest, dest_offset);
//     return status;
// }

// Ands 2 1bit objects into a 3rd 1bit object (associated)
// All PimObjId parameters and offsets should be replaced with 1bit pim objects
PimStatus pimAnd1bit(PimObjId src1, uint64_t src1_offset, PimObjId src2, uint64_t src2_offset, PimObjId dest, uint64_t dest_offset) {

    PimStatus status;
    status = pimOpReadRowToSa(src1, src1_offset);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpMove(src1, PIM_RREG_SA, PIM_RREG_R1);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpReadRowToSa(src2, src2_offset);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpAnd(src1, PIM_RREG_SA, PIM_RREG_R1, PIM_RREG_SA);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpWriteSaToRow(dest, dest_offset);
    return status;
}

// Inverts a 1bit pim object into another 1bit object (associated)
// All PimObjId parameters and offsets should be replaced with 1bit pim objects
PimStatus pimNot1bit(PimObjId src, uint64_t src_offset, PimObjId dest, uint64_t dest_offset) {

    PimStatus status;
    status = pimOpReadRowToSa(src, src_offset);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpNot(src, PIM_RREG_SA, PIM_RREG_SA);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpWriteSaToRow(dest, dest_offset);
    return status;
}

// Sets 0th bit in 32bit object to 0th bit in 1bit object (must be associated with each other)
// Sets all other bits to 0
// src, src_offset would become a 1bit pim object
// dest stays as a 32bit pim object
PimStatus pimCast1BitTo32Bit(PimObjId src, uint64_t src_offset, PimObjId dest) {

    PimStatus status;

    status = pimBroadcastUInt(dest, 0);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpReadRowToSa(src, src_offset);
    if(status != PIM_OK) {
        return status;
    }

    status = pimOpWriteSaToRow(dest, 0);
    return status;
}

#endif