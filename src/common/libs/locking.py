# -*- coding: utf-8 -*-
from pathlib import Path
import os
import random
import time
import unittest

LOCKDIR = os.path.join(Path.home(), "locks")
LOCK_FPATH = {"cpu": os.path.join(LOCKDIR, "cpu"), "gpu": os.path.join(LOCKDIR, "gpu")}
BACKOFF_SECONDS = 1
os.makedirs(LOCKDIR, exist_ok=True)


def is_locked(device="gpu"):
    try:
        with open(LOCK_FPATH[device], "r") as f:
            # if (lock_owner := int(f.readline())) == os.getpid():  # not compat w/ python 3.6
            lock_owner = int(f.readline())
            if lock_owner == os.getpid():
                return False
            try:
                os.kill(lock_owner, 0)  # harmless
            except OSError:  # PID doesn't exist
                return False
            return True
    except FileNotFoundError:
        return False


def is_owned(device="gpu"):
    try:
        with open(LOCK_FPATH[device], "r") as f:
            if int(f.readline()) == os.getpid():
                return True
            return False
    except FileNotFoundError:
        return False


def lock(device: str = "gpu"):
    """device: gpu, cpu"""
    backoff_time = BACKOFF_SECONDS
    while not is_owned(device):
        if is_locked(device):
            time.sleep(int(backoff_time))
            backoff_time = min(10 + random.random(), backoff_time * 1.1)
            if (
                backoff_time > BACKOFF_SECONDS * 1.1**9
                and backoff_time < BACKOFF_SECONDS * 1.1**11
            ):
                print("lock: spinning for %s..." % device)
        else:
            with open(LOCK_FPATH[device], "w") as f:
                f.write(str(os.getpid()))
            if backoff_time >= BACKOFF_SECONDS * 1.1**9:
                print("lock:ed %s." % device)


def unlock(device: str = "gpu"):
    """device: gpu, cpu"""
    if is_owned(device):
        os.remove(LOCK_FPATH[device])
        return True
    return False


def check_pause():
    """
    touch ~/locking/pause_<PID>
    to pause
    """
    backoff_time = BACKOFF_SECONDS
    lock_fpath = os.path.join(LOCKDIR, "pause_%u" % os.getpid())
    lock_fpath_universal = os.path.join(LOCKDIR, "pause_all")
    while os.path.isfile(lock_fpath) or os.path.isfile(lock_fpath_universal):
        time.sleep(backoff_time)
        backoff_time += 1
        if backoff_time == BACKOFF_SECONDS + 10:
            print(f"paused by {lock_fpath} or {lock_fpath_universal}")


class Test_locking(unittest.TestCase):
    def test_lock_unlock(self):
        lock("cpu")
        self.assertTrue(is_owned("cpu"))
        self.assertFalse(is_owned("gpu"))
        lock("gpu")
        self.assertTrue(is_owned("gpu"))
        self.assertFalse(is_locked("gpu"))
        self.assertTrue(unlock("gpu"))
        with open(LOCK_FPATH["cpu"], "w") as f:
            f.write("123")
        self.assertFalse(unlock("cpu"))
        self.assertFalse(is_locked("cpu"))
        self.assertFalse(is_owned("cpu"))
        os.remove(LOCK_FPATH["cpu"])


if __name__ == "__main__":
    unittest.main()
