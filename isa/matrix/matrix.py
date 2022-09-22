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
        if 'a' == self['dim']:
            if self['mltr'] == "true":
                ret = self["rs1"][0:self["tilek"], 0:self["tilem"]].transpose()
            else:
                ret = self["rs1"][0:self["tilem"], 0:self["tilek"]]
        elif 'b' == self['dim']:
            if self['mrtr'] == "true":
                ret = self["rs1"][0:self["tilen"], 0:self["tilek"]].transpose()
            else:
                ret = self["rs1"][0:self["tilek"], 0:self["tilen"]]
        elif 'c' == self['dim']:
            ret = self["rs1"][0:self["tilem"], 0:self["tilen"]]
        
        return ret


class Mse_test(Inst):
    def golden(self):
        stride = self["stride"] // (self["eew"] // 8)
        if "tr" in self["tt"]:
            ret = np.zeros(shape=(128*stride), dtype=self["rs1"].dtype)
        else:
            ret = np.zeros(shape=(self["mlen"] // self["vlen"]*stride), dtype=self["rs1"].dtype)

        if "tr" in self["tt"] :
            if self["dim"] == 'a' :
                if self["mltr"] == "true":
                    height = self["tilek"]
                    width = self["tilem"]
                else:
                    height = self["tilem"]
                    width = self["tilek"]
            elif self["dim"] == 'b' :
                if self["mrtr"] == "true":
                    height = self["tilen"]
                    width = self["tilek"]
                else:
                    height = self["tilek"]
                    width = self["tilen"]
        else:
            height = self["tilem"]
            width = self["tilen"]

        if 'c' in self["tt"]:
            for i in range(height):
                ret[i*stride :  i*stride + width] = self["rs1"][0:width, i]
        else:
            for i in range(height):
                ret[i*stride :  i*stride + width] = self["rs1"][i, 0:width]
        
        return ret
        
def get_out_type(insn, stype):
    if insn == "mopa" or insn == "mfopa":
        return stype
    elif insn == "mwopa" or insn == "mfwopa":
        if stype == np.int8:
            return np.int16
        elif stype == np.int16:
            return np.int32
        elif stype == np.int32:
            return np.int64
        elif stype == np.float16:
            return np.float32
    elif insn == "mqopa":
        if stype == np.int8:
            return np.int32

class Mopa_test(Inst):
    def golden(self):
        ret_type = get_out_type(self["mopa"], self["rs1"].dtype)
        if "q" in self["mopa"]:
            ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * 4//self["eew"]), dtype=ret_type)
        elif "w" in self["mopa"]:
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


class Madd_test(Inst):
    def golden(self):
        factor = 4 // (self["eew"] // self["sew"])
        ret = np.zeros(shape=(self["mlen"]//self["vlen"], self["vlen"] * factor //self["sew"]), dtype=self["rs1"].dtype)
        if 'f' in self["madd"]:
            result_type = np.float128
        else:
            result_type = np.int64
        
        if "add" in self["madd"]:
            result = np.add(self["rd"].astype(result_type), self["rs1"].astype(result_type), dtype=result_type)
        elif "rsub" in self["madd"]:
            result = np.add(self["rd"].astype(result_type), -(self["rs1"].astype(result_type)), dtype=result_type)
        elif "sub" in self["madd"]:
            result = np.add(-(self["rd"].astype(result_type)), self["rs1"].astype(result_type), dtype=result_type)

        if 'f' in self["madd"]:
            result = np.where(result > np.finfo(self["rs1"].dtype).max, np.finfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.finfo(self["rs1"].dtype).min, np.finfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        else:
            result = np.where(result > np.iinfo(self["rs1"].dtype).max, np.iinfo(self["rs1"].dtype).max, result)
            result = np.where(result < np.iinfo(self["rs1"].dtype).min, np.iinfo(self["rs1"].dtype).min, result).astype(self["rs1"].dtype)
        
        ret[0:self["tilem"], 0:self["tilen"]] = result
        return ret
