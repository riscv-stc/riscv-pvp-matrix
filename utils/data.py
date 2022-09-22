'''You can use functions and lists in this module to generate data for cases.
'''

import numpy as np
import jax.numpy as jnp

## matrix
def get_tile_max(mlen, vlen, dim, msew):
    if dim == 'm':
        return mlen // vlen
    elif dim == 'n':
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

def get_mload_tt(dim, mltr, mrtr):
    if dim == 'a' :
        if mltr == "true":
            return ["tr.c"]
        else:
            return ["tr.r"]
    elif dim == 'b':
        if mrtr == 'true':
            return ["tr.c"]
        else:
            return ["tr.r"]
    elif dim == 'c':
         return ["xa.r"]

def get_mload_data(dim, eew, tilem, tilen, tilek, stride_byte, mltr='false', mrtr='false'):
    stride_ele = stride_byte // (eew // 8)
    if dim == "c":
        return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "a":
        if mltr == "true":
            return np.random.randint(2^eew, size=(tilek, stride_ele), dtype=bits_to_dtype_int(eew))
        else:
            return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))
    elif dim == "b":
        if mrtr == "true":
            return np.random.randint(2^eew, size=(tilen, stride_ele), dtype=bits_to_dtype_int(eew))
        else:
            return np.random.randint(2^eew, size=(tilek, stride_ele), dtype=bits_to_dtype_int(eew))
    else:
        return np.random.randint(2^eew, size=(tilem, stride_ele), dtype=bits_to_dtype_int(eew))

def get_mload_stride(dim, eew, tilem, tilen, tilek, mltr="false", mrtr="false"):
    if dim == 'a':
        if mltr == "true":
            stride = tilem * eew // 8
        else:
            stride =  tilek * eew // 8
    elif dim == 'b' :
        if mrtr == "true":
            stride =  tilek * eew // 8
        else:
            stride =  tilen * eew // 8
    elif dim == 'c' :
        stride =  tilen * eew // 8
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


def get_mopa_sew(insn):
    if insn == "mfwopa":
        return [16]
    elif insn == "mfopa":
        return [16, 32]
    elif insn == "mqopa":
        return [8]
    elif insn == "mwopa":
        return [8, 16]
    elif insn == "mopa":
        return [8, 16, 32]
    else:
        return [16]

def get_mopa_eew(mopa, sew):
    if 'q' in mopa:
        return 4*sew
    elif 'w' in mopa:
        return 2*sew
    else:
        return sew

def get_mopa_src1(insn, eew, tilem, tilek):
    if 'f' in insn:
        return np.random.random((tilem, tilek)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-2, 3, size=(tilem, tilek), dtype=bits_to_dtype_int(eew))

def get_mopa_src2(insn, eew, tilek, tilen):
    if 'f' in insn:
        return np.random.random((tilek, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-2, 3, size=(tilek, tilen), dtype=bits_to_dtype_int(eew))

def get_mopa_mmv(mopa):
    if 'q' in mopa:
        return "mqmv"
    elif 'w' in mopa:
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

def get_madd_sew(insn):
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

def get_madd_eew(insn, sew):
    if "w" in insn:
        return 2*sew
    elif "q" in insn:
        return 4*sew
    else:
        return sew


def get_madd_src(insn, eew, tilem, tilen):
    if 'f' in insn:
        return np.random.random((tilem, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(10, size=(tilem, tilen), dtype=bits_to_dtype_int(eew))


def get_random_src(insn, eew, tilem, tilen):
    if 'f' in insn:
        return np.random.random((tilem, tilen)).astype(bits_to_dtype_float(eew)) * 2 - 1
    else:
        return np.random.randint(-pow(2, eew-1), pow(2, eew-1), size=(tilem, tilen), dtype=bits_to_dtype_int(eew))