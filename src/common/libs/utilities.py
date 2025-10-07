# -*- coding: utf-8 -*-
"""Common utility functions and classes for file handling, data processing, and system operations.

This module provides a collection of utility functions used throughout the project,
including file operations, data serialization/deserialization, path manipulation,
multithreading helpers, compression utilities, and data structure operations.

Key functional areas:
- File operations: checksum, cp, backup, filesize
- Date and time utilities: get_date
- Multithreading: mt_runner for parallel processing
- Data serialization: JSON, YAML, and pickle read/write functions
- Directory and path manipulation: get_leaf, get_root, get_file_dname
- Compression utilities: compress_lzma, compress_png, decompress_lzma
- Data structure manipulation: freeze_dict, unfreeze_dict, shuffle_dictionary
- Logging and printing: Printer class

Most functions are designed to be simple, focused helpers that perform specific
tasks with error handling appropriate for an academic research codebase.
"""

import os
import sys
import logging
import random
from typing import Callable, Union, Iterable, Optional, List, Any
import tqdm
import json
import csv
import lzma
import shutil
import datetime
import unittest
import pickle
import atexit  # restart_program()

# Use torch.multiprocessing for proper tensor sharing
try:
    import torch
    import torch.multiprocessing as mp
    from torch.multiprocessing import Pool

    TORCH_AVAILABLE = True
except ImportError:
    from multiprocessing import Pool
    import multiprocessing as mp

    TORCH_AVAILABLE = False


try:
    import png
except ModuleNotFoundError as e:
    logging.error(f"{e} (install pypng)")
import numpy as np
import statistics
import subprocess
import hashlib
import yaml

# sys.path += ['..', '.']
NUM_THREADS = os.cpu_count()


def checksum(fpath, htype="sha1"):
    """Calculate the cryptographic hash of a file.

    Computes a hash digest of the file's contents using the specified hash algorithm.
    Useful for verifying file integrity or identifying duplicate files.

    Args:
        fpath: Path to the file to hash
        htype: Hash algorithm to use ("sha1" or "sha256")

    Returns:
        Hexadecimal digest string of the file's hash

    Raises:
        NotImplementedError: If an unsupported hash type is specified
        FileNotFoundError: If the file does not exist
    """
    if htype == "sha1":
        h = hashlib.sha1()
    elif htype == "sha256":
        h = hashlib.sha256()
    else:
        raise NotImplementedError(type)
    with open(fpath, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cp(inpath, outpath, verbose=False, overwrite=True):
    """Copy a file with optional verbose output and overwrite control.

    Attempts to use copy-on-write (via --reflink=auto) when available, falling back
    to standard copy if not supported. This is more efficient on filesystems that
    support CoW (like Btrfs, XFS).

    Args:
        inpath: Source file path to copy from
        outpath: Destination file path to copy to
        verbose: If True, print a message showing the copy operation
        overwrite: If False, append "dupath.ext" suffix when destination exists
                  rather than overwriting
    """
    if not overwrite:
        while os.path.isfile(outpath):
            outpath = outpath + "dupath." + outpath.split(".")[-1]
    try:
        subprocess.run(("cp", "--reflink=auto", inpath, outpath))
    except FileNotFoundError:
        shutil.copy2(inpath, outpath)
    if verbose:
        print(f"cp {inpath} {outpath}")


def get_date() -> str:
    """Get the current date in ISO format (YYYY-MM-DD).

    Returns:
        String containing the current date in YYYY-MM-DD format
    """
    return f"{datetime.datetime.now():%Y-%m-%d}"


def backup(filepaths: list):
    """Backup files with date-stamped filenames.

    Creates a 'backup' directory in the current working directory if it doesn't exist,
    then copies each specified file into that directory with the current date
    prepended to the filename.

    Args:
        filepaths: List of file paths to backup

    Note:
        Backup filenames have format: YYYY-MM-DD_original_filename
    """
    if not os.path.isdir("backup"):
        os.makedirs("backup", exist_ok=True)
    date = get_date()
    for fpath in filepaths:
        fn = get_leaf(fpath)
        shutil.copy(fpath, os.path.join("backup", date + "_" + fn))


def mt_runner(
    fun: Callable[[Any], Any],
    argslist: list,
    num_threads: int = NUM_THREADS,
    ordered: bool = False,
    progress_bar: bool = True,
    starmap: bool = False,
    progress_desc: str = "Processing",
) -> Iterable[Any]:
    """Execute a function in parallel across multiple processes with progress tracking.

    This is a general-purpose multiprocessing wrapper that handles process pool management,
    progress visualization, and proper cleanup. It uses torch.multiprocessing when available
    for CUDA compatibility, falling back to standard multiprocessing otherwise.

    The function automatically sets the spawn start method for CUDA safety and provides
    both ordered (slower but preserves sequence) and unordered (faster) execution modes.
    Single-threaded execution is supported for debugging.

    Args:
        fun: Function to execute in parallel (should accept single argument unless starmap=True)
        argslist: List of arguments to pass to the function (one per task)
        num_threads: Number of worker processes (defaults to CPU count)
        ordered: If True, preserve input order in results (slower due to waiting for stragglers)
        progress_bar: If True, display a tqdm progress bar with statistics
        starmap: If True, unpack each argument tuple as *args to the function
                (requires ordered=True)
        progress_desc: Text description shown in the progress bar

    Returns:
        List of results in the same order as argslist (if ordered=True) or completion order

    Note:
        When verbose mode is detected in argslist, the progress bar displays scene and
        method information extracted from results (useful for dataset processing tasks).
    """
    if num_threads is None:
        num_threads = NUM_THREADS

    # Check if verbose flag is set in argslist
    verbose = False
    if argslist and len(argslist) > 0:
        if isinstance(argslist[0], dict) and "verbose" in argslist[0]:
            verbose = argslist[0]["verbose"]

    # Set multiprocessing start method to spawn for CUDA compatibility
    if TORCH_AVAILABLE:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method already set
            pass

    if num_threads == 1:
        # Single-threaded execution
        results = []
        for args in argslist:
            if starmap:
                results.append(fun(*args))
            else:
                results.append(fun(args))
        return results

    # Use context manager for automatic cleanup of pool resources
    with Pool(num_threads) as pool:
        try:
            if starmap:
                amap = pool.starmap
                if not ordered:
                    raise NotImplementedError("Unordered starmap")
            elif ordered:
                amap = pool.imap
            else:
                amap = pool.imap_unordered

            if ordered:
                print("mt_runner warning: ordered=True might be slower.")
                if progress_bar:
                    print(
                        "mt_runner warning: progress bar NotImplemented for ordered pool."
                    )
                ret = list(amap(fun, argslist))
            else:
                if progress_bar:
                    ret = []
                    try:
                        # Use a stationary progress bar that updates in place
                        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        pbar = tqdm.tqdm(
                            amap(fun, argslist),
                            total=len(argslist),
                            desc=progress_desc,
                            bar_format=bar_format,
                            position=0,
                            leave=True,
                            dynamic_ncols=True,
                            file=sys.stderr,
                            mininterval=0.05,
                        )

                        current_scene = None
                        current_method = None

                        for i, ares in enumerate(pbar):
                            ret.append(ares)
                            # Extract scene and method info for display (only in verbose mode)
                            if verbose and hasattr(ares, "get") and "gt_fpath" in ares:
                                # Extract scene and actual method used from the result
                                scene_name = ares.get("image_set", "unknown")
                                method = ares.get("alignment_method", "unknown")

                                # Only update description if scene or method changed to avoid flicker
                                if (
                                    scene_name != current_scene
                                    or method != current_method
                                ):
                                    current_scene = scene_name
                                    current_method = method
                                    desc = f"Scene: {scene_name:<30} Method: {method.upper()}"
                                    pbar.set_description(desc)

                        # Don't close the progress bar - leave it on screen showing final state
                        pbar.close()

                    except (TypeError, KeyboardInterrupt) as e:
                        if isinstance(e, KeyboardInterrupt):
                            logging.info("Multiprocessing interrupted by user")
                        else:
                            logging.error(f"Progress bar error: {e}")
                        # Re-raise to trigger cleanup
                        raise
                else:
                    ret = list(amap(fun, argslist))

        except Exception as e:
            logging.error(f"Multiprocessing error: {e}")
            # Context manager will handle cleanup
            raise

        return ret


def jsonfpath_load(fpath, default_type=dict, default=None):
    if not os.path.isfile(fpath):
        print(
            "jsonfpath_load: warning: {} does not exist, returning default".format(
                fpath
            )
        )
        if default is None:
            return default_type()
        else:
            return default

    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {k if not k.isdigit() else int(k): v for k, v in x.items()}
        return x

    with open(fpath, "r") as f:
        return json.load(f, object_hook=jsonKeys2int)


def jsonfpath_to_dict(fpath):
    print("warning: jsonfpath_to_dict is deprecated, use jsonfpath_load instead")
    return jsonfpath_load(fpath, default_type=dict)


def dict_to_json(adict, fpath):
    with open(fpath, "w") as f:
        json.dump(adict, f, indent=2)


def dict_to_yaml(adict, fpath):
    """Serialize a dictionary to a YAML file.

    Args:
        adict: Dictionary to serialize
        fpath: Output file path for the YAML file
    """
    with open(fpath, "w") as f:
        yaml.dump(adict, f, allow_unicode=True)


def load_yaml(
    fpath: str, safely=True, default_type=dict, default=None, error_on_404=True
):
    """Load and parse a YAML file with automatic key type conversion.

    Reads a YAML file and automatically converts string-digit keys to integers
    (e.g., "42" becomes 42). Handles missing files gracefully with configurable
    default return values.

    Args:
        fpath: Path to the YAML file
        safely: If True, use yaml.safe_load (recommended); if False, use yaml.load
        default_type: Type to instantiate for default return value (dict or list)
        default: Explicit default value to return if file doesn't exist
        error_on_404: If False, return default instead of raising FileNotFoundError

    Returns:
        Parsed YAML content (typically dict or list) with integer keys where applicable
    """
    if not os.path.isfile(fpath) and not error_on_404:
        print(
            "jsonfpath_load: warning: {} does not exist, returning default".format(
                fpath
            )
        )
        if default is None:
            return default_type()
        else:
            return default
    with open(fpath, "r") as f:
        if safely:
            res = yaml.safe_load(f)
        else:
            res = yaml.load(f, Loader=yaml.Loader)
    # transform string number keys into int
    if isinstance(res, dict):
        keys_to_convert = []
        for akey in res.keys():
            if isinstance(akey, str) and akey.isdigit():
                keys_to_convert.append(akey)
        for akey in keys_to_convert:
            res[int(akey)] = res[akey]
            del res[akey]
    return res


def dict_to_pickle(adict, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(adict, f)


def picklefpath_to_dict(fpath):
    with open(fpath, "rb") as f:
        adict = pickle.load(f)
    return adict


def args_to_file(fpath):
    with open(fpath, "w") as f:
        f.write("python " + " ".join(sys.argv))


def save_listofdict_to_csv(listofdict, fpath, keys=None, mixed_keys=False):
    """
    Use mixed_keys=True if different dict have different keys.
    """
    if keys is None:
        keys = listofdict[0].keys()
        if mixed_keys:
            keys = set(keys)
            for somekeys in [adict.keys() for adict in listofdict]:
                keys.update(somekeys)
    keys = sorted(keys)
    try:
        with open(fpath, "w", newline="") as f:
            csvwriter = csv.DictWriter(f, keys)
            csvwriter.writeheader()
            csvwriter.writerows(listofdict)
    except ValueError as e:
        print(
            "save_listofdict_to_csv: error: {}. This likely means that the dictionaries have different keys, try passing mixed_keys=True".format(
                e
            )
        )
        breakpoint()


class Printer:
    """Simple dual-output logger that writes to both stdout and a file.

    Provides a convenient interface for logging messages to both console and a
    persistent log file simultaneously. Useful for experiments where you want
    both interactive feedback and a permanent record.

    Args:
        tostdout: If True, print messages to standard output
        tofile: If True, append messages to log file
        save_dir: Directory for the log file (created if doesn't exist)
        fn: Filename for the log file (without extension)
        save_file_path: Override full path to log file (ignores save_dir and fn if set)
    """

    def __init__(
        self, tostdout=True, tofile=True, save_dir=".", fn="log", save_file_path=None
    ):
        self.tostdout = tostdout
        self.tofile = tofile
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = (
            os.path.join(save_dir, fn) if save_file_path is None else save_file_path
        )

    def print(self, msg, err=False):  # TODO to stderr if err
        if self.tostdout:
            print(msg)
        if self.tofile:
            try:
                with open(self.file_path, "a") as f:
                    f.write(str(msg) + "\n")
            except Exception as e:
                print("Warning: could not write to log: %s" % e)


def std_bpp(bpp) -> str:
    try:
        return "{:.2f}".format(float(bpp))
    except TypeError:
        return None


def get_leaf(path: str) -> str:
    """Extract the final component (filename or directory name) from a path.

    Works correctly regardless of trailing slashes on directory paths.

    Args:
        path: File or directory path

    Returns:
        The leaf name (final path component)
    """
    return os.path.basename(os.path.relpath(path))


def get_root(fpath: str) -> str:
    """
    return root directory a file (fpath) is located in.
    """
    while fpath.endswith(os.pathsep):
        fpath = fpath[:-1]
    return os.path.dirname(fpath)


def get_file_dname(fpath: str) -> str:
    return os.path.basename(os.path.dirname(fpath))


def freeze_dict(adict: dict) -> frozenset:
    """Recursively convert a dictionary into a hashable frozenset representation.

    Useful for using dictionaries as dictionary keys or in sets. Nested dictionaries
    are recursively frozen as well.

    Args:
        adict: Dictionary to freeze

    Returns:
        Frozen representation as a frozenset of key-value tuples
    """
    fdict = adict.copy()
    for akey, aval in fdict.items():
        if isinstance(aval, dict):
            fdict[akey] = freeze_dict(aval)
    return frozenset(fdict.items())


def unfreeze_dict(fdict: frozenset) -> dict:
    adict = dict(fdict)
    for akey, aval in adict.items():
        if isinstance(aval, frozenset):
            adict[akey] = unfreeze_dict(aval)
    return adict


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


def dict_of_frozendicts2csv(res, fpath):
    """dict of frozendicts to csv
    used in eg evolve/tools/test_weights_on_all_tasks"""
    reslist = []
    dkeys = set()
    for areskey, aresval in res.items():
        ares = dict()
        for componentkey, componentres in unfreeze_dict(areskey).items():
            if isinstance(componentres, dict):
                for subcomponentkey, subcomponentres in componentres.items():
                    ares[componentkey + "_" + subcomponentkey] = subcomponentres
            else:
                ares[componentkey] = componentres
        ares["res"] = aresval
        reslist.append(ares)
        dkeys.update(ares.keys())
    save_listofdict_to_csv(reslist, fpath, dkeys)


def list_of_tuples_to_csv(listoftuples, heading, fpath):
    with open(fpath, "w") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(heading)
        for arow in listoftuples:
            csvwriter.writerow(arow)


def dpath_has_content(dpath: str):
    if not os.path.isdir(dpath):
        return False
    return len(os.listdir(dpath)) > 0


def str2gp(gpstr):
    """Convert str(((gains), (priorities))) to tuple(((gains), (priorities)))"""
    # print(tuple([tuple([int(el) for el in weights.split(', ')]) for weights in gpstr[2:-2].split('), (')])) # dbg
    try:
        return tuple(
            [
                tuple([int(el) for el in weights.split(", ")])
                for weights in gpstr[2:-2].split("), (")
            ]
        )
    except ValueError:
        breakpoint()


def get_highest_direntry(dpath: str) -> Optional[str]:
    """Get highest numbered entry in a directory"""
    highest = -1
    for adir in os.listdir(dpath):
        if adir.isdecimal() and int(adir) > highest:
            highest = int(adir)
    if highest == -1:
        return None
    return str(highest)


def get_last_modified_file(
    dpath,
    exclude: Optional[Union[str, List[str]]] = None,
    incl_ext: bool = True,
    full_path=True,
    fn_beginswith: Optional[Union[str, int]] = None,
    ext=None,
    exclude_ext: Optional[str] = None,
):
    """Get the last modified fn,
    optionally excluding patterns found in exclude (str or list),
    optionally omitting extension"""
    if not os.path.isdir(dpath):
        return False
    fpaths = [
        os.path.join(dpath, fn) for fn in os.listdir(dpath)
    ]  # add path to each file
    fpaths.sort(key=os.path.getmtime, reverse=True)
    if len(fpaths) == 0:
        return False
    fpath = None
    if exclude is None and fn_beginswith is None and ext is None:
        fpath = fpaths[0]
    else:
        if isinstance(exclude, str):
            exclude = [exclude]
        if isinstance(fn_beginswith, int):
            fn_beginswith = str(fn_beginswith)
        for afpath in fpaths:
            fn = afpath.split("/")[-1]  # not Windows friendly
            if exclude is not None and fn in exclude:
                continue
            if fn_beginswith is not None and not fn.startswith(fn_beginswith):
                continue
            if ext is not None and not fn.endswith("." + ext):
                continue
            if exclude_ext is not None and fn.endswith("." + exclude_ext):
                continue
            fpath = afpath
            break
        if fpath is None:
            return False
    if not incl_ext:
        assert "." in fpath.split("/")[-1], fpath  # not Windows friendly
        fpath = fpath.rpartition(".")[0]
    if full_path:
        return fpath
    else:
        return fpath.split("/")[-1]


def listfpaths(dpath):
    """Similar to os.listdir(dpath), returns joined paths of files present."""
    fpaths = []
    for fn in os.listdir(dpath):
        fpaths.append(os.path.join(dpath, fn))
    return fpaths


def compress_lzma(infpath, outfpath):
    with open(infpath, "rb") as f:
        dat = f.read()
    # DBG: timing lzma compression
    # tic = time.perf_counter()
    cdat = lzma.compress(dat)
    # toc = time.perf_counter()-tic
    # print("compress_lzma: side_string encoding time = {}".format(toc))
    # compress_lzma: side_string encoding time = 0.005527787026949227
    # tic = time.perf_counter()
    # ddat = lzma.decompress(dat)
    # toc = time.perf_counter()-tic
    # print("compress_lzma: side_string decoding time = {}".format(toc))

    #
    with open(outfpath, "wb") as f:
        f.write(cdat)


def compress_png(tensor, outfpath):
    """only supports grayscale!"""
    if tensor.shape[0] > 1:
        print("common.utilities.compress_png: warning: too many channels (failed)")
        return False
    w = png.Writer(
        tensor.shape[2],
        tensor.shape[1],
        greyscale=True,
        bitdepth=int(np.ceil(np.log2(tensor.max() + 1))),
        compression=9,
    )
    with open(outfpath, "wb") as fp:
        w.write(fp, tensor[0])
    return True


def decompress_lzma(infpath, outfpath):
    with open(infpath, "rb") as f:
        cdat = f.read()
    dat = lzma.decompress(cdat)
    with open(outfpath, "wb") as f:
        f.write(dat)


# def csv_fpath_to_listofdicts(fpath):
# TODO parse int/float
#     with open(fpath, 'r') as fp:
#         csvres = list(csv.DictReader(fp))
#     return csvres


# def save_src(root_dpath: str, directories: list[str], extensions: list[str] = ["py"]):
#    pass  # TODO for dn in directories:


def noop(*args, **kwargs):
    pass


def filesize(fpath):
    return os.stat(fpath).st_size


def avg_listofdicts(listofdicts):
    res = dict()
    for akey in listofdicts[0].keys():
        res[akey] = list()
    for adict in listofdicts:
        for akey, aval in adict.items():
            res[akey].append(aval)
    for akey in res.keys():
        res[akey] = statistics.mean(res[akey])
    return res


def walk(root: str, dir: str = ".", follow_links=False):
    """Similar to os.walk, but keeps a constant root"""
    dpath = os.path.join(root, dir)
    for name in os.listdir(dpath):
        path = os.path.join(dpath, name)
        if os.path.isfile(path):
            yield (root, dir, name)
        elif os.path.isdir(path) or (os.path.islink(path) and follow_links):
            yield from walk(root=root, dir=os.path.join(dir, name))
        elif os.path.islink(path) and not follow_links:
            continue
        elif not os.path.exists(path):
            print(f"walk: {path=} disappeared, ignoring.")
            continue
        else:
            # raise ValueError(f"Unknown type: {path}")
            popup(f"walk: Unknown type: {path}")
            breakpoint()


def popup(msg):
    """Print and send a notification on Linux/compatible systems."""
    print(msg)
    subprocess.run(["/usr/bin/notify-send", msg], check=False)


def restart_program():
    def _restart_program():
        os.execl(sys.executable, sys.executable, *sys.argv)

    atexit.register(_restart_program)
    exit()


def shuffle_dictionary(input_dict):
    # Convert the dictionary to a list of key-value pairs
    items = list(input_dict.items())

    # Shuffle the list of items
    random.shuffle(items)

    # Convert the shuffled list back to a dictionary
    shuffled_dict = dict(items)

    return shuffled_dict


def sort_dictionary(input_dict):
    # Convert the dictionary to a list of key-value pairs
    items = list(input_dict.items())

    # Sort the list of items
    items.sort()

    # Convert the sorted list back to a dictionary
    sorted_dict = dict(items)

    return sorted_dict


class Test_utilities(unittest.TestCase):
    def test_freezedict(self):
        adict = {"a": 1, "b": 22, "c": 333, "d": {"e": 4, "f": 555}}
        print(adict)
        fdict = freeze_dict(adict)
        print(fdict)
        ndict = {fdict: 42}
        adictuf = unfreeze_dict(fdict)
        print(adictuf)
        self.assertDictEqual(adict, adictuf)


if __name__ == "__main__":
    unittest.main()
