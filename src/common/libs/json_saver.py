"""JSONSaver class makes for a simple way to save/load train/test results."""

from typing import Set
import sys
import os

sys.path.append("..")
from common.libs import utilities


class JSONSaver:
    def __init__(
        self,
        jsonfpath,
        step_type: str = ["step", "epoch"][0],
        default=None,
        warmup_nsteps: int = 0,
    ):
        assert not os.path.isdir(jsonfpath)
        if default is None:
            default = {"best_val": dict()}
        self.best_key_str = "best_{}".format(step_type)  # best step/epoch #
        self.jsonfpath = jsonfpath
        self.results = self._load(jsonfpath, default=default)
        if self.best_key_str not in self.results:
            self.results[self.best_key_str] = dict()
        self.warmup_nsteps = warmup_nsteps

    def _load(self, fpath, default):
        return utilities.jsonfpath_load(fpath, default=default)

    def add_res(
        self,
        step: int,
        res: dict,
        minimize=True,
        write=True,
        val_type=float,
        epoch=None,
        rm_none=False,
        key_prefix="",
    ):
        """epoch is an alias for step
        Set rm_none to True to ignore zero values"""
        if epoch is not None and step is None:
            step = epoch
        elif (epoch is None and step is None) or step is None or epoch is not None:
            raise ValueError("JSONSaver.add_res: Must specify either step or epoch")
        if step not in self.results:
            self.results[step] = dict()
        if key_prefix != "":
            res_ = dict()
            for akey, aval in res.items():
                res_[key_prefix + akey] = aval
            res = res_
        for akey, aval in res.items():
            if aval is None:
                print(f"JSONSaver.add_res warning: minnig value for {akey}")
                continue
            if val_type is not None:
                aval = val_type(aval)
            self.results[step][akey] = aval
            if isinstance(aval, list):
                continue
            if rm_none and aval == 0:
                continue
            # check for best_step
            if step < self.warmup_nsteps:
                continue
            # if akey not in self.results[self.best_key_str]:
            #     self.results[self.best_key_str][akey] = step
            #     self.results["best_val"][akey] = aval
            if akey not in self.results["best_val"]:
                self.results["best_val"][akey] = aval
            if akey not in self.results[self.best_key_str]:
                self.results[self.best_key_str][akey] = step
            if (self.results["best_val"][akey] > aval and minimize) or (
                self.results["best_val"][akey] < aval and not minimize
            ):
                self.results[self.best_key_str][akey] = step
                self.results["best_val"][akey] = aval
        if write:
            self.write()

    def write(self):
        utilities.dict_to_json(self.results, self.jsonfpath)

    def get_best_steps(self) -> Set[int]:
        return set(self.results[self.best_key_str].values())

    def get_best_step(self, akey) -> int:
        return self.results[self.best_key_str][akey]

    def get_best_step_results(self, akey) -> dict:
        return self.results[self.get_best_step(akey)]

    def is_empty(self) -> bool:
        """Returns True if there are no saved results."""
        return len(self.results["best_val"]) == 0


class YAMLSaver(JSONSaver):
    def __init__(
        self,
        jsonfpath,
        step_type: str = ["step", "epoch"][0],
        default=None,
        warmup_nsteps: int = 0,
    ):
        super().__init__(
            jsonfpath, step_type=step_type, default=default, warmup_nsteps=warmup_nsteps
        )

    def _load(self, fpath, default):
        return utilities.load_yaml(fpath, default=default, error_on_404=False)

    def write(self):
        utilities.dict_to_yaml(self.results, self.jsonfpath)
