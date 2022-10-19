'''You can use functions and lists in this module to generate data for cases.
'''

import numpy as np
import jax.numpy as jnp

## matrix
def get_tile_max(mlen, vlen, dim, msew):
    if 'm' in dim:
        return mlen // vlen
    elif 'n' in dim:
        return vlen // msew
    else:
        return min(mlen//vlen, vlen//msew)

def bits_to_dtype_int(sew):
    '''Function to get int data type corresponding the input width.

    Args:
        sew (int): vsew register value, int data width, which can be 8, 16, 32, 64.

    Returns:
        dtype: numpy int dtype corresponding to the data width, numpy.int8 corresponding to 8,
        numpy.int16 corresponding to 16, numpy.int32 corresponding to 32, numpy.int64 corresponding to 64.
    '''
    int_dtype_dict = { 8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64 }
    return int_dtype_dict[sew]

def bits_to_dtype_float(sew):
    int_dtype_dict = { 16: np.float16, 32: np.float32, 64: np.float64}
    return int_dtype_dict[sew]

def get_mload_data(dim, eew, tilem, tilen, tilek, stride_byte):
    stride_ele = stride_byte // (eew // 8)
    if dim == "ct":
        return np.random.randint(2^eew, size=(tilen, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "c":
        return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "at":
        return np.random.randint(2^eew, size=(tilek, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "a":
        return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "bt":
        return np.random.randint(2^eew, size=(tilen, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "b":
        return np.random.randint(2^eew, size=(tilek, stride_ele), dtype=bits_to_dtype_int(eew))
    else:
        return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))

def get_mstore_data(dim, height, width, eew) :
    return np.random.randint(2^eew, size=(height, width), dtype=bits_to_dtype_int(eew))


def get_mload_stride(dim, eew, tilem, tilen, tilek):
    if dim == 'at':
        stride = tilem * eew // 8
    elif dim == 'a':
        stride =  tilek * eew // 8
    elif dim == 'bt' :
        stride =  tilek * eew // 8
    elif dim == 'b':
        stride =  tilen * eew // 8
    elif dim == 'ct' :
        stride = tilem * eew // 8
    elif dim == 'c' :
        stride = tilen * eew // 8
    else :
        return 0
    return [stride, stride + eew // 8, stride * 2]

def get_mload_len(tile, mlen, vlen, sew):
    hmax = mlen // vlen
    wmax = vlen // sew
    if tile == "tilek": # tilek
        tmax = min(hmax, wmax)
    elif tile == "tilem" : # tilem
        tmax =  hmax
    elif tile == "tilen" : # tilen
        tmax = wmax
    else :
        return [1]
    return [1, tmax//2, tmax]

def get_mmv_len(tt, dim, mlen, vlen, sew):
    mmax = mlen // vlen
    nmax = vlen // sew
    kmax = min(mmax, nmax)
    if "c" in tt:
        if dim == "k": # tilek
            tmax = min(kmax, mmax)
        elif dim == "m" : # tilem
            tmax =  min(mmax, nmax)
        elif dim == "n" : # tilen
            tmax = min(nmax, mmax)
    else :
        if dim == "k": # tilek
            tmax = kmax
        elif dim == "m" : # tilem
            tmax =  min(mmax, nmax)
        elif dim == "n" : # tilen
            tmax = nmax
    return [1, tmax//2, tmax]

def get_mmv_slice(tt, mlen, vlen, sew):
    mmax = mlen // vlen
    nmax = vlen // sew
    if "c" in tt:
        slice_max = nmax
    else :
        slice_max = mmax
    return [0, slice_max//2-1, slice_max-1]


def get_mma_sew(insn):
    if insn == "mfwma":
        return [16]
    elif insn == "mfma":
        return [16, 32]
    elif insn == "mqma":
        return [8]
    elif insn == "mwma":
        return [8, 16]
    elif insn == "mma":
        return [8, 16, 32]
    else:
        return [16]

def get_mma_eew(mma, sew):
    if 'q' in mma:
        return 4*sew
    elif 'w' in mma:
        return 2*sew
    else:
        return sew

def get_mma_src1(insn, eew, tilem, tilek) :
    if 'f' in insn:
        return np.random.random((tilem, tilek)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-2, 3, size=(tilem, tilek), dtype=bits_to_dtype_int(eew))

def get_mma_src2(insn, eew, tilek, tilen):
    if 'f' in insn:
        return np.random.random((tilek, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-2, 3, size=(tilek, tilen), dtype=bits_to_dtype_int(eew))

def get_mopa_mmv(mma):
    if 'q' in mma:
        return "mqmv"
    elif 'w' in mo:
        return "mwmv"
    else:
        return "mmv" 

def get_mfopa_mmv(mopa):
    if 'q' in mopa:
        return "mfqmv"
    elif 'w' in mopa:
        return "mfwmv"
    else:
        return "mfmv" 

def get_acc_load_eew(sew):
    if sew == 8 :
        return [8, 16, 32]
    elif sew == 16:
        return [16, 32]
    elif sew == 32:
        return [32]
    else:
        return [32]

def get_acc_fload_eew(sew):
    if sew == 16:
        return [16, 32]
    elif sew == 32:
        return [32]
    else:
        return [32]

def get_acc_mmv(sew, eew):
    factor = eew // sew
    if factor == 1:
        return "mmv"
    elif factor == 2:
        return "mwmv"
    elif factor == 4:
        return "mqmv"
    else:
        return "mqmv"

def get_acc_mfmv(sew, eew):
    factor = eew // sew
    if factor == 1:
        return "mfmv"
    elif factor == 2:
        return "mfwmv"
    elif factor == 4:
        return "mfqmv"
    else:
        return "mfqmv" 

def get_maddc_sew(insn):
    if "fw" in insn:
        return [16]
    elif "f" in insn:
        return [16, 32]
    elif "w" in insn:
        return [8, 16]
    elif "q" in insn:
        return [8]
    else:
        return [8, 16, 32]

def get_maddc_eew(insn, sew):
    if "w" in insn:
        return 2*sew
    elif "q" in insn:
        return 4*sew
    else:
        return sew


def get_maddc_src(insn, eew, tilem, tilen):
    if 'f' in insn:
        return np.random.random((tilem, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(10, size=(tilem, tilen), dtype=bits_to_dtype_int(eew))


def get_random_src(insn, eew, tilem, tilen):
    if 'f' in insn:
        return np.random.random((tilem, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-pow(2, eew-1), pow(2, eew-1), size=(tilem, tilen), dtype=bits_to_dtype_int(eew))


def get_memulc_sew(insn):
    if "fw" in insn:
        return [16]
    elif "f" in insn:
        return [16, 32]
    elif "w" in insn:
        return [8, 16]
    elif "q" in insn:
        return [8]
    else:
        return [8, 16, 32]

def get_memulc_eew(insn, sew):
    if "w" in insn:
        return 2*sew
    elif "q" in insn:
        return 4*sew
    else:
        return sew

def get_memulc_ld(eew):
    if eew == 8:
        return "lb"
    elif eew == 16:
        return "lh"
    elif eew == 32:
        return "lw"
    elif eew == 64:
        return "ld"

def get_memulc_fld(eew):
    if eew == 16:
        return "flh"
    elif eew == 32:
        return "flw"
    elif eew == 64:
        return "fld"
        

def get_memulc_src1(insn, eew, tilem, tilen):
    if 'f' in insn:
        return np.random.random((tilem, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(10, size=(tilem, tilen), dtype=bits_to_dtype_int(eew))

def get_memulc_src2(insn, eew, num):
    ret = []
    for i in range(num):
        if 'f' in insn:
            ret.append(np.random.random(1).astype(bits_to_dtype_float(eew))*2-1)
        else:
            ret.append(np.random.randint(0, 2^eew, 1, dtype=bits_to_dtype_int(eew)))

    return ret


def get_mfcvt_sew(insn):
    insn_name = insn.split('.')[0]
    if 'w' in insn_name or 'n' in insn_name:
        return [16]
    else:
        return [16, 32]

def get_mfcvt_seew(insn, sew):
    insn_src = insn.split('.')[2]
    if 'q' in insn_src:
        return 4*sew
    elif 'w' in insn_src:
        return 2*sew
    else:
        return sew

def get_mfcvt_deew(insn, sew):
    insn_dst = insn.split('.')[1]
    if 'q' in insn_dst:
        return 4*sew
    elif 'w' in insn_dst:
        return 2*sew
    else:
        return sew

def get_mfcvt_src(insn, seew, tilem, tilen):
    insn_src = insn.split('.')[2]
    if 'f' in insn_src:
        return np.random.randint(0, 2^seew, (tilem, tilen)).astype(bits_to_dtype_float(seew))
    else:
        return np.random.randint(0, 2^seew, (tilem, tilen)).astype(bits_to_dtype_int(seew))