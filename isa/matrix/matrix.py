from unittest import result
from rvpvp.isa.inst import *
import numpy as np
import jax.numpy as jnp

class Mtype_test(Inst):
    def golden(self):
        pass

class Msettilemi_test(Inst):
    def golden(self):
        pass

class Mle_test(Inst):
    def golden(self):
        if self['dim'] == 'at' :
            ret = self["rs1"][0:self["tilek"], 0:self["tilem"]]
        elif self['dim'] == 'a' :
            ret = self["rs1"][0:self["tilem"], 0:self["tilek"]]
        elif self['dim'] == 'bt':
            ret = self["rs1"][0:self["tilen"], 0:self["tilek"]]
        elif self['dim'] == 'b':
            ret = self["rs1"][0:self["tilek"], 0:self["tilen"]]
        elif self['dim'] == 'ct' :
            ret =  self["rs1"][0:self["tilen"], 0:self["tilem"]]
        elif self['dim'] == 'c':
            ret = self["rs1"][0:self["tilem"], 0:self["tilen"]]
        
        return ret


class Mse_test(Inst):
    def golden(self):
        stride = self["stride"] // (self["eew"] // 8)
        ret = np.zeros(shape=(128*stride), dtype=self["rs1"].dtype)
            

        if self["dim"] == 'at' :
            height = self["tilek"]
            width = self["tilem"]
        elif self["dim"] == 'a' :
            height = self["tilem"]
            width = self["tilek"]
        elif self["dim"] == 'bt' :
            height = self["tilen"]
            width = self["tilek"]
        elif self["dim"] == 'b':
            height = self["tilek"]
            width = self["tilen"]
        elif self["dim"] == 'ct':
            height = self["tilen"]
            width = self["tilem"]
        elif self["dim"] == 'c':
            height = self["tilem"]
            width = self["tilen"]

    
        if 't' in self["dim"]:
            for i in range(height):
                ret[i*stride :  i*stride + width] = self["rs1"][0:width, i]
        else:
            for i in range(height):
                ret[i*stride :  i*stride + width] = self["rs1"][i, 0:width]

        
        return ret
        
def get_out_type(insn, stype):
    if insn == "mma" or insn == "mfma":
        return stype
    elif insn == "mwma" or insn == "mfwma":
        if stype == np.int8:
            return np.int16
        elif stype == np.int16:
            return np.int32
        elif stype == np.int32:
            return np.int64
        elif stype == np.float16:
            return np.float32
    elif insn == "mqma":
        if stype == np.int8:
            return np.int32

class Mma_test(Inst):
    def golden(self):
        ret_type = get_out_type(self["mma"], self["rs1"].dtype)
        if "q" in self["mma"]:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * 4//self["eew"]), dtype=ret_type)
        elif "w" in self["mma"]:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * 4// self["eew"]), dtype=ret_type)
        else:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * 4//self["eew"]), dtype=ret_type)

        out = np.matmul(self["rs1"].astype(ret_type), self["rs2"].astype(ret_type)).astype(ret_type)
        ret[0:self["tilem"], 0:self["tilen"]] = out
        return ret


class Mmv_out_test(Inst):
    def golden(self):
        ret = np.zeros(shape=(self['rs1'].shape[1]), dtype=self['rs1'].dtype)

        if 'c' in self['tt']:
            ret[0:self['tile_len']] = self['rs1'][0:self['tile_len'], self['slice']]
        else:
            ret[0:self['tile_len']] = self['rs1'][self['slice'], 0:self['tile_len']]

        return ret


class Mmv_in_test(Inst):
    def golden(self):
        ret = np.zeros(shape=(self['mlen']//self['vlen'], self['vlen']//self['sew']), dtype=self['rs1'].dtype)
        if 'm' in self['dim']:
            ret = np.zeros(shape=(self['mlen']//self['vlen'], self['vlen']*4 //(self['rs1'].itemsize * 8)), dtype=self['rs1'].dtype)
        

        if 'c' in self['tt']:
            ret[0:self['tile_len'], self['slice']] = self['rs1'][0:self['tile_len']]
        else:
            ret[self['slice'], 0:self['tile_len']] = self['rs1'][0:self['tile_len']]

        return ret


class Maddc_test(Inst):
    def golden(self):
        factor = 4 // (self["eew"] // self["sew"])
        ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * factor //self["sew"]), dtype=self["rs1"].dtype)
        if 'f' in self["maddc"]:
            result_type = np.float128
        else:
            result_type = np.int64
        
        if "addc" in self["maddc"]:
            result = np.add(self["rd"].astype(result_type), self["rs1"].astype(result_type), dtype=result_type)
        elif "rsubc" in self["maddc"]:
            result = np.add(-(self["rd"].astype(result_type)), self["rs1"].astype(result_type), dtype=result_type)
        elif "subc" in self["maddc"]:
            result = np.add(self["rd"].astype(result_type), -(self["rs1"].astype(result_type)), dtype=result_type)
            

        if 'f' in self["maddc"]:
            result = np.where(result > np.finfo(self["rs1"].dtype).max, np.finfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.finfo(self["rs1"].dtype).min, np.finfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        else:
            result = np.where(result > np.iinfo(self["rs1"].dtype).max, np.iinfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.iinfo(self["rs1"].dtype).min, np.iinfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        
        ret[0:self["tilem"], 0:self["tilen"]] = result
        return ret


class Memulc_test(Inst):
    def golden(self):
        factor = 4 // (self["eew"] // self["sew"])
        ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * factor //self["sew"]), dtype=self["rs1"].dtype)
        if 'f' in self["memulc"]:
            result_type = np.float128
        else:
            result_type = np.int64

        result = self["rs1"].astype(result_type) * self["rs2"]

        if 'f' in self["memulc"]:
            result = np.where(result > np.finfo(self["rs1"].dtype).max, np.finfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.finfo(self["rs1"].dtype).min, np.finfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        else:
            result = np.where(result > np.iinfo(self["rs1"].dtype).max, np.iinfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.iinfo(self["rs1"].dtype).min, np.iinfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        
        ret[0:self["tilem"], 0:self["tilen"]] = result
        return ret

def bits_to_dtype_float(sew):
    int_dtype_dict = { 16: np.float16, 32: np.float32, 64: np.float64}
    return int_dtype_dict[sew]

def bits_to_dtype_int(sew):
    int_dtype_dict = { 8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64 }
    return int_dtype_dict[sew]

class Mfcvt_test(Inst):
    def golden(self):
        insn_name = self["mfcvt"].split('.')[0]
        insn_dst = self["mfcvt"].split('.')[1]
        insn_src = self["mfcvt"].split('.')[2]
        if 'f' in insn_dst:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"]*4//self["deew"]), dtype=bits_to_dtype_float(self["deew"]))
            ret[0:self["tilem"], 0:self["tilen"]] = self["rs1"].astype(bits_to_dtype_float(self["deew"]))
        else:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"]*4//self["deew"]), dtype=bits_to_dtype_int(self["deew"]))
            ret[0:self["tilem"], 0:self["tilen"]] = self["rs1"].astype(bits_to_dtype_int(self["deew"]))

        return ret
        
        
