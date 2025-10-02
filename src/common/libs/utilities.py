# -*- coding: utf-8 -*-
"""Common utilities."""

import os
import sys
import logging
import random
from typing import Callable, Union, Iterable, Optional, List, Any
import asyncio
import threading
import queue
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
import time
import numpy as np
import statistics
import subprocess
import hashlib
import yaml

# sys.path += ['..', '.']
NUM_THREADS = os.cpu_count()


class GPUMemoryScheduler:
    """Memory-aware GPU task scheduler that prevents OOM errors with cross-process coordination."""
    
    def __init__(self, shared_state=None):
        self.available_memory = None
        self.memory_lock = threading.Lock()
        self.pending_tasks = queue.Queue()
        self.active_tasks = {}
        self._initialized = False
        self.shared_state = shared_state  # Shared across processes
        self._tensor_buffers = {}  # Buffer reuse pool for tensors
    
    def _initialize_gpu_memory(self):
        """Initialize GPU memory tracking (called lazily to avoid fork issues)."""
        if self._initialized or not TORCH_AVAILABLE:
            return
            
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                # Reserve 20% for system and fragmentation
                self.available_memory = int(total_memory * 0.8)
                
                # Initialize shared state if available
                if self.shared_state is not None:
                    with self.shared_state.lock:
                        if not self.shared_state['initialized']:
                            self.shared_state['total_memory'] = self.available_memory
                            self.shared_state['allocated_memory'] = 0
                            self.shared_state['initialized'] = True
                
                # Only log once per process to avoid spam
                if not hasattr(self, '_logged_init'):
                    logging.info(f"GPU memory scheduler initialized: {self.available_memory / 1e9:.1f}GB available (PID: {os.getpid()})")
                    self._logged_init = True
            else:
                self.available_memory = 0
                if not hasattr(self, '_logged_init'):
                    logging.info("No CUDA available, GPU memory scheduler disabled")
                    self._logged_init = True
        except Exception as e:
            if not hasattr(self, '_logged_init'):
                logging.warning(f"Failed to initialize GPU memory scheduler: {e}")
                self._logged_init = True
            self.available_memory = 0
        
        self._initialized = True
    
    def get_tensor_buffer(self, shape: tuple, dtype=None, device=None) -> 'torch.Tensor':
        """Get a reusable tensor buffer to avoid memory allocations."""
        if not TORCH_AVAILABLE:
            return None
            
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device('cpu')
            
        buffer_key = (shape, dtype, device)
        
        # Try to reuse existing buffer
        if buffer_key in self._tensor_buffers and len(self._tensor_buffers[buffer_key]) > 0:
            return self._tensor_buffers[buffer_key].pop()
        
        # Create new buffer
        try:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            if device.type == 'cpu':
                buffer.share_memory_()  # Enable sharing for CPU tensors
            return buffer
        except Exception as e:
            logging.debug(f"Failed to create tensor buffer {shape} on {device}: {e}")
            return None
    
    def return_tensor_buffer(self, tensor: 'torch.Tensor'):
        """Return a tensor buffer to the pool for reuse."""
        if not TORCH_AVAILABLE or tensor is None:
            return
            
        buffer_key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        
        if buffer_key not in self._tensor_buffers:
            self._tensor_buffers[buffer_key] = []
        
        # Limit buffer pool size to prevent memory bloat
        if len(self._tensor_buffers[buffer_key]) < 3:
            self._tensor_buffers[buffer_key].append(tensor)
    
    def estimate_memory_usage(self, height: int, width: int, channels: int = 3, dtype_size: int = 4) -> int:
        """Estimate GPU memory usage for image processing operations.
        
        Args:
            height, width, channels: Image dimensions
            dtype_size: Bytes per element (4 for float32)
            
        Returns:
            Estimated memory usage in bytes
        """
        # Base memory for 2 input images
        base_memory = 2 * height * width * channels * dtype_size
        
        # FFT operations typically need 4-6x the input size for intermediates
        # Add padding, complex numbers, and workspace
        estimated_memory = base_memory * 6
        
        return estimated_memory
    
    def can_acquire_memory(self, required_memory: int) -> bool:
        """Check if enough GPU memory is available."""
        if not self._initialized:
            self._initialize_gpu_memory()
            
        if self.available_memory is None or self.available_memory == 0:
            return False
            
        with self.memory_lock:
            return self.available_memory >= required_memory
    
    def acquire_memory(self, task_id: str, required_memory: int, timeout: float = 300.0) -> bool:
        """Acquire GPU memory for a task, waiting if necessary.
        
        Args:
            task_id: Unique identifier for the task
            required_memory: Memory required in bytes
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if memory was acquired, False if timeout or no GPU
        """
        if not self._initialized:
            self._initialize_gpu_memory()
            
        if self.available_memory is None or self.available_memory == 0:
            return False
        
        # If memory requirement is too large, reject immediately
        if required_memory > self.available_memory + sum(self.active_tasks.values()):
            logging.warning(f"Task {task_id} requires {required_memory / 1e9:.1f}GB but GPU only has "
                          f"{(self.available_memory + sum(self.active_tasks.values())) / 1e9:.1f}GB total")
            return False
            
        start_time = time.time()
        wait_logged = False
        
        while time.time() - start_time < timeout:
            # Use shared state if available, otherwise fall back to local tracking
            if self.shared_state is not None:
                with self.shared_state.lock:
                    available = self.shared_state['total_memory'] - self.shared_state['allocated_memory']
                    if available >= required_memory:
                        self.shared_state['allocated_memory'] += required_memory
                        self.active_tasks[task_id] = required_memory
                        if wait_logged:
                            logging.info(f"Task {task_id} acquired {required_memory / 1e6:.1f}MB GPU memory after waiting")
                        else:
                            logging.debug(f"Acquired {required_memory / 1e6:.1f}MB GPU memory for task {task_id}, "
                                        f"{available - required_memory / 1e6:.1f}MB remaining")
                        return True
                    current_available = available
            else:
                with self.memory_lock:
                    if self.available_memory >= required_memory:
                        self.available_memory -= required_memory
                        self.active_tasks[task_id] = required_memory
                        if wait_logged:
                            logging.info(f"Task {task_id} acquired {required_memory / 1e6:.1f}MB GPU memory after waiting")
                        else:
                            logging.debug(f"Acquired {required_memory / 1e6:.1f}MB GPU memory for task {task_id}, "
                                        f"{self.available_memory / 1e6:.1f}MB remaining")
                        return True
                    current_available = self.available_memory
            
            # Log waiting message once
            if not wait_logged:
                logging.info(f"Task {task_id} waiting for {required_memory / 1e6:.1f}MB GPU memory "
                           f"({current_available / 1e6:.1f}MB available)")
                wait_logged = True
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        logging.warning(f"Task {task_id} timed out waiting for GPU memory after {timeout}s")
        return False
    
    def release_memory(self, task_id: str):
        """Release GPU memory for a completed task."""
        if task_id in self.active_tasks:
            memory_amount = self.active_tasks.pop(task_id)
            
            # Update shared state if available
            if self.shared_state is not None:
                with self.shared_state.lock:
                    self.shared_state['allocated_memory'] -= memory_amount
                    available = self.shared_state['total_memory'] - self.shared_state['allocated_memory']
                    logging.debug(f"Released {memory_amount / 1e6:.1f}MB GPU memory for task {task_id}, "
                                f"{available / 1e6:.1f}MB available")
            else:
                with self.memory_lock:
                    self.available_memory += memory_amount
                    logging.debug(f"Released {memory_amount / 1e6:.1f}MB GPU memory for task {task_id}, "
                                f"{self.available_memory / 1e6:.1f}MB available")


# Shared GPU memory state across processes
_shared_gpu_memory = None
_gpu_scheduler = None
_scheduler_lock = threading.Lock()


def _initialize_shared_gpu_memory():
    """Initialize shared GPU memory tracking across processes."""
    global _shared_gpu_memory
    if _shared_gpu_memory is None and TORCH_AVAILABLE:
        try:
            # Create shared memory for cross-process coordination
            manager = mp.Manager()
            _shared_gpu_memory = manager.dict({
                'total_memory': 0,
                'allocated_memory': 0,
                'initialized': False
            })
            _shared_gpu_memory.lock = manager.Lock()
            logging.info("Shared GPU memory state initialized")
        except Exception as e:
            logging.warning(f"Failed to create shared GPU memory state: {e}")
            _shared_gpu_memory = None
    return _shared_gpu_memory


def _get_shared_gpu_memory():
    """Get the shared GPU memory state, initializing if needed."""
    global _shared_gpu_memory
    if _shared_gpu_memory is None:
        _shared_gpu_memory = _initialize_shared_gpu_memory()
    return _shared_gpu_memory


def get_gpu_scheduler() -> GPUMemoryScheduler:
    """Get the process-local GPU memory scheduler instance with shared coordination."""
    global _gpu_scheduler
    if _gpu_scheduler is None:
        with _scheduler_lock:
            if _gpu_scheduler is None:
                shared_state = _get_shared_gpu_memory()
                _gpu_scheduler = GPUMemoryScheduler(shared_state)
    return _gpu_scheduler


class GPUAwareWorkerWrapper:
    """Picklable wrapper that adds GPU memory management to worker functions."""
    
    def __init__(self, func: Callable, estimate_memory_func: Optional[Callable] = None, num_processes: int = 1):
        self.func = func
        self.estimate_memory_func = estimate_memory_func
        self.num_processes = num_processes
    
    def __call__(self, args):
        # Set CPU thread count to avoid oversubscription
        if TORCH_AVAILABLE:
            # Calculate threads per process: total_cores / num_processes
            threads_per_process = max(1, NUM_THREADS // self.num_processes)
            torch.set_num_threads(threads_per_process)
            logging.debug(f"Set torch threads to {threads_per_process} for worker process {os.getpid()} "
                         f"({NUM_THREADS} total cores / {self.num_processes} processes)")
        
        # Handle GPU memory scheduling if estimate function provided
        task_id = None
        if self.estimate_memory_func and TORCH_AVAILABLE:
            try:
                task_id, memory_estimate = self.estimate_memory_func(args)
                scheduler = get_gpu_scheduler()
                
                # Try to acquire GPU memory
                if scheduler.acquire_memory(task_id, memory_estimate):
                    try:
                        result = self.func(args)
                        return result
                    finally:
                        scheduler.release_memory(task_id)
                else:
                    # GPU memory not available, function should handle fallback
                    logging.debug(f"GPU memory not available for task {task_id}, function will handle fallback")
                    result = self.func(args)
                    return result
            except Exception as e:
                if task_id:
                    get_gpu_scheduler().release_memory(task_id)
                logging.error(f"Error in GPU-aware worker: {e}")
                raise
        else:
            # No GPU memory management, just run the function
            return self.func(args)


def gpu_aware_worker_wrapper(func: Callable, estimate_memory_func: Optional[Callable] = None, num_processes: int = 1):
    """Create a picklable wrapper that adds GPU memory management to worker functions.
    
    Args:
        func: The worker function to wrap
        estimate_memory_func: Function to estimate memory usage from args
                             Should return (task_id, memory_estimate) tuple
        num_processes: Number of worker processes (for CPU thread limiting)
    
    Returns:
        Wrapped function that handles GPU memory scheduling
    """
    return GPUAwareWorkerWrapper(func, estimate_memory_func, num_processes)


def checksum(fpath, htype="sha1"):
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
    return f"{datetime.datetime.now():%Y-%m-%d}"


def backup(filepaths: list):
    """Backup a given list of files per day"""
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
    gpu_memory_estimator: Optional[Callable] = None,
) -> Iterable[Any]:
    """
    Multiprocessing runner with GPU memory management and proper tensor sharing.
    
    Args:
        fun: function to run
        argslist: list of arguments to pass to function
        num_threads: number of worker processes
        ordered: whether to preserve order (slower)
        progress_bar: show progress bar
        starmap: expand arguments (not compatible with ordered=False)
        progress_desc: description for progress bar
        gpu_memory_estimator: function to estimate GPU memory usage from args
                             Should return (task_id, memory_estimate) tuple
    """
    if num_threads is None:
        num_threads = NUM_THREADS
    
    # Set multiprocessing start method to spawn for CUDA compatibility
    if TORCH_AVAILABLE:
        try:
            mp.set_start_method('spawn', force=True)
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
    else:
        # Initialize shared GPU memory state in main process
        global _shared_gpu_memory
        if _shared_gpu_memory is None:
            _shared_gpu_memory = _initialize_shared_gpu_memory()
        
        # Wrap function with GPU memory management if estimator provided
        if gpu_memory_estimator:
            wrapped_fun = gpu_aware_worker_wrapper(fun, gpu_memory_estimator, num_threads)
        else:
            wrapped_fun = gpu_aware_worker_wrapper(fun, None, num_threads)
        
        pool = Pool(num_threads)
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
                ret = list(amap(wrapped_fun, argslist))
            else:
                if progress_bar:
                    ret = []
                    try:
                        # Use a stationary progress bar that sticks to bottom
                        bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                        pbar = tqdm.tqdm(amap(wrapped_fun, argslist), total=len(argslist), desc=progress_desc, 
                                       bar_format=bar_format, position=0, leave=False, ncols=120, 
                                       dynamic_ncols=False, file=sys.stdout, mininterval=0.1)
                        
                        current_scene = None
                        current_method = None
                        
                        for i, ares in enumerate(pbar):
                            ret.append(ares)
                            # Extract scene and method info for display
                            if hasattr(ares, 'get') and 'gt_fpath' in ares:
                                # Extract scene from the result
                                scene_name = ares.get('image_set', 'unknown')
                                method = ares.get('alignment_method', 'auto')
                                
                                # Only update description if scene or method changed to avoid flicker
                                if scene_name != current_scene or method != current_method:
                                    current_scene = scene_name
                                    current_method = method
                                    desc = f"Scene: {scene_name:<30} Method: {method.upper()}"
                                    pbar.set_description(desc)
                        
                        # Clear the progress bar after completion to avoid leaving it on screen
                        pbar.close()
                                
                    except (TypeError, KeyboardInterrupt) as e:
                        if isinstance(e, KeyboardInterrupt):
                            logging.info("Multiprocessing interrupted by user")
                        else:
                            logging.error(f"Progress bar error: {e}")
                        # Re-raise to trigger cleanup
                        raise
                else:
                    ret = list(amap(wrapped_fun, argslist))
                    
        except Exception as e:
            logging.error(f"Multiprocessing error: {e}")
            # Terminate all worker processes immediately
            pool.terminate()
            pool.join()
            raise
        else:
            # Normal cleanup - close pool and wait for workers to finish
            pool.close()
            pool.join()
            
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
    with open(fpath, "w") as f:
        yaml.dump(adict, f, allow_unicode=True)


def load_yaml(
    fpath: str, safely=True, default_type=dict, default=None, error_on_404=True
):
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
    """Returns the leaf of a path, whether it's a file or directory followed by
    / or not."""
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
    """Recursively freeze a dictionary into hashable type"""
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
