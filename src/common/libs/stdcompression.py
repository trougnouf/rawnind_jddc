# -*- coding: utf-8 -*-
"""Standard compression methods handlers."""
import os
from typing import List, Any, Optional
import shutil
import unittest
import subprocess
import inspect
import time
import sys

sys.path.append("..")
from common.libs import utilities

NUMTHREADS: int = 1  # os.cpu_count()//4*3
CHROMA_SS: List[int] = [444]
VALID_ARGS: list = ["quality", "chroma_ss", "bitrate", "weights", "profile"]
LOSSLESS_IMGEXT = ["png", "ppm"]


class StdCompression:
    def __init__(self):
        assert shutil.which(self.BINARY) is not None, "Missing {} binary".format(
            self.ENCBIN
        )

    #    @classmethod:
    #    def make_comp_fpath(cls, infpath: str, compfpath: str):
    #        breakpoint()

    @classmethod
    def make_tmp_fpath(cls, outfpath: str, tmpfpath: Optional[str] = None):
        if not cls.REQ_DEC:
            assert tmpfpath is None or outfpath == tmpfpath
            return outfpath
        if tmpfpath is not None:
            return tmpfpath
        if cls.REQ_DEC:
            return outfpath + "." + cls.ENCEXT

    @classmethod
    def file_encdec(
        cls,
        infpath: str,
        outfpath: Optional[str],
        tmpfpath: Optional[str] = None,
        cleanup: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):

        assert outfpath is not None
        tmpfpath = cls.make_tmp_fpath(outfpath, tmpfpath)
        returnvals = {"infpath": infpath, "outfpath": outfpath, "tmpfpath": tmpfpath}
        assert set.issubset(set(kwargs.keys()), set(VALID_ARGS))
        assert shutil.which(cls.ENCBIN) is not None, "Missing {} binary".format(
            cls.ENCBIN
        )
        assert tmpfpath.endswith(cls.ENCEXT), tmpfpath
        assert outfpath.split(".")[-1] in LOSSLESS_IMGEXT or not cls.REQ_DEC, outfpath
        cmd = cls.make_enc_cl(infpath, tmpfpath, **kwargs)
        if not os.path.isfile(tmpfpath) or overwrite:
            start_time = time.time()
            subprocess.run(cmd)
            returnvals["enctime"] = time.time() - start_time
        assert os.path.isfile(tmpfpath), f'{tmpfpath=}, {" ".join(cmd)}'
        returnvals["encsize"] = os.path.getsize(tmpfpath)
        if tmpfpath != outfpath:
            cmd = cls.make_dec_cl(tmpfpath, outfpath, **kwargs)
            if not os.path.isfile(outfpath) or overwrite:
                # print(' '.join(cmd))
                start_time = time.time()
                subprocess.run(cmd)
                returnvals["dectime"] = time.time() - start_time
            if cleanup:
                os.remove(tmpfpath)
        # print(" ".join(cmd))  # DBG
        return returnvals

    @classmethod
    def make_cname(cls, **kwargs):
        cname: str = cls.__name__ + str(kwargs)
        cname = cname.replace(" ", "_")
        cname = cname.replace("'", "")
        return cname

    @classmethod
    def file_encdec_mtrunner(cls, kwargs):
        return cls.file_encdec(**kwargs)

    @classmethod
    def dir_encdec(
        cls, indpath: str, outdpath: str, cleanup=True, overwrite=False, **kwargs
    ):
        imgs = os.listdir(indpath)
        args = []
        if outdpath is None:
            dsname = utilities.get_leaf(indpath)
            outdpath = os.path.join(
                indpath, "compressed", cls.make_cname(kwargs), dsname
            )
        os.makedirs(outdpath, exist_ok=True)
        for fn in imgs:
            outfpath = os.path.join(outdpath, fn)
            if not cls.REQ_DEC:
                if outfpath.split(".")[-1] != cls.ENCEXT:
                    outfpath = outfpath + "." + cls.ENCEXT
            elif outfpath.split(".")[-1] not in LOSSLESS_IMGEXT:
                outfpath = outfpath + "." + LOSSLESS_IMGEXT[0]
            if os.path.isfile(outfpath) and not overwrite:
                continue
            args.append(
                {
                    "infpath": os.path.join(indpath, fn),
                    "outfpath": outfpath,
                    "cleanup": cleanup,
                    **kwargs,
                }
            )
        utilities.mt_runner(
            cls.file_encdec_mtrunner, args, num_threads=NUMTHREADS, ordered=False
        )
        return outdpath


class JPG_Compression(StdCompression):

    ENCBIN = BINARY = "gm"
    ENCEXT: str = "jpg"
    REQ_DEC: bool = False
    QUALITY_RANGE = (1, 100 + 1)

    @classmethod
    def make_enc_cl(cls, infpath: str, outfpath: str, quality: Any, **kwargs) -> list:
        assert quality is not None
        return [
            JPG_Compression.BINARY,
            "convert",
            infpath,
            "-strip",
            "-quality",
            "{}%".format(quality),
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        return [JPG_Compression.BINARY, "convert", infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class JPEGXL_Compression(StdCompression):

    ENCBIN = BINARY = "cjxl"
    DECBIN = "djxl"
    ENCEXT: str = "jxl"
    REQ_DEC: bool = True
    QUALITY_RANGE = (1, 100 + 1)

    @classmethod
    def make_enc_cl(cls, infpath: str, outfpath: str, quality: Any, **kwargs) -> list:
        assert quality is not None
        return [
            cls.BINARY,
            "--quality",
            str(quality),
            infpath,
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        return [cls.DECBIN, infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class BPG_Compression(StdCompression):

    ENCBIN: str = "bpgenc"
    DECBIN: str = "bpgdec"
    ENCEXT: str = "bpg"
    REQ_DEC: bool = True
    QUALITY_RANGE = (0, 51 + 1)

    @classmethod
    def make_enc_cl(
        cls,
        infpath: str,
        outfpath: str,
        quality: Any,
        chroma_ss: Any = CHROMA_SS[0],
        **kwargs,
    ) -> list:
        assert quality is not None
        return [
            BPG_Compression.ENCBIN,
            "-q",
            str(quality),
            "-f",
            str(chroma_ss),
            "-o",
            outfpath,
            infpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        return [BPG_Compression.DECBIN, infpath, "-o", outfpath]

    @classmethod
    def get_valid_cargs(cls):
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class JPEGXS_Compression(StdCompression):
    ENCBIN: str = "tco_encoder"
    DECBIN: str = "tco_decoder"
    ENCEXT: str = "tco"
    REQ_DEC: bool = True
    PROFILE = 11
    WEIGHTS = ["psnr", "visual"]
    BITRATE_RANGE = (0.36, 1.51, 0.01)

    @classmethod
    def make_enc_cl(
        cls,
        infpath: str,
        outfpath: str,
        bitrate: Any,
        profile=PROFILE,
        weights: str = WEIGHTS[0],
        **kwargs,
    ) -> list:
        assert bitrate is not None
        return [
            cls.ENCBIN,
            "-b",
            str(bitrate),
            "-p",
            str(profile),
            "-o",
            weights,
            infpath,
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        return [cls.DECBIN, infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        bitrate = cls.BITRATE_RANGE[0]
        while bitrate < cls.BITRATE_RANGE[1]:
            for weights in cls.WEIGHTS:
                yield {"bitrate": bitrate, "weights": weights, "profile": 11}
            bitrate += cls.BITRATE_RANGE[2]


COMPRESSIONS = [
    acls[0] for acls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
]
COMPRESSIONS.remove("StdCompression")


class Test_utilities(unittest.TestCase):
    def test_compress_kodak_bpg(self):
        indpath = os.path.join("..", "..", "datasets", "test", "kodak")
        cname = BPG_Compression.make_cname(quality=50)
        outdpath = os.path.join(
            "..", "..", "datasets", "test", "compressed", cname, "kodak"
        )
        os.makedirs(outdpath, exist_ok=True)
        BPG_Compression.dir_encdec(
            indpath=indpath, outdpath=outdpath, quality=50, cleanup=True
        )
        self.assertGreater(len(os.listdir(outdpath)), 0)

    def test_compress_kodak_jpg(self):
        indpath = os.path.join("..", "..", "datasets", "test", "kodak")
        cname = JPG_Compression.make_cname(quality=50)
        outdpath = os.path.join(
            "..", "..", "datasets", "test", "compressed", cname, "kodak"
        )
        os.makedirs(outdpath, exist_ok=True)
        JPG_Compression.dir_encdec(indpath=indpath, outdpath=outdpath, quality=50)
        self.assertGreater(len(os.listdir(outdpath)), 0)


if __name__ == "__main__":
    unittest.main()
