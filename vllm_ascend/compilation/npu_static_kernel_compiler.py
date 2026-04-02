
import hashlib
import os
import subprocess
from pathlib import Path
import shutil
import time
import torch_npu
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_world_group
from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import logger

from vllm.config.utils import hash_factors

def common_hash_factors(vllm_config, compilation_config):
    env_factors = envs.compile_factors()
    env_hash = hash_factors(env_factors)
    config_hash = vllm_config.compute_hash()

    forward_code_files = list(sorted(compilation_config.traced_files))
    hash_content = []
    for filepath in forward_code_files:
        hash_content.append(filepath)
        if filepath == "<string>":
            continue
        try:
            with open(filepath) as f:
                hash_content.append(f.read())
        except Exception:
            logger.warning("Failed to read file %s", filepath)
            continue
    code_hash = hashlib.sha512("\n".join(hash_content).encode()).hexdigest()
    factors = [env_hash, config_hash, code_hash]
    return factors

class StaticKernelCompiler:
    def __init__(self, vllm_config, static_kernels):
        self.static_kernels = [kernel.lower() for kernel in static_kernels]
        self.disable_static_kernels = [kernel.lower() for kernel in static_kernels if kernel.startswith('-')]
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0))
        self.is_global_first_rank = is_global_first_rank() if local_world_size <=0 else \
            get_world_group().rank_in_group % local_world_size == 0
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        factors = common_hash_factors(vllm_config=vllm_config, compilation_config=vllm_config.compilation_config)
        hash_key = hashlib.sha512(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
        self.cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "static_kernel_cache", hash_key)
        local_cache_dir = os.path.join(self.cache_dir, f"rank_{rank}_{dp_rank}")
        os.makedirs(local_cache_dir, exist_ok=True)
        self.local_cache_root = Path(local_cache_dir)
        self.cache_root = Path(self.cache_dir)
        self.latest_cache_dir = os.path.join(self.cache_dir, f"merge")

        if os.path.exists(self.cache_dir) and any(self.cache_root.glob("*.run")):
            if self.is_global_first_rank:
                logger.info(f"Found static kernel package in directory: %s, skipping compilation",
                            self.cache_dir)
            self.cache_hit = True
        else:
            logger.info(f"Using directory: %s for static kernel compile", local_cache_dir)
            self.cache_hit = False
            if os.path.exists(self.latest_cache_dir):
                shutil.rmtree(self.latest_cache_dir, ignore_errors=True)
            os.makedirs(self.latest_cache_dir, exist_ok=True)


    def __enter__(self):
        if "+uninstall" in self.static_kernels:
            if self.is_global_first_rank:
                self.uninstall()
            self.reselect_static_kernel()
            return self
        
        if self.cache_hit:
            return self
        
        logger.debug("Starting operator dump...")
        import acl
        acl.op.start_dump_args(1, str(self.local_cache_root))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if "+uninstall" in self.static_kernels:
            return

        if self.cache_hit:
            return

        import acl
        acl.op.stop_dump_args(1)
        logger.debug("Stopping operator dump.")

        if get_world_group().world_size > 1:
            get_world_group().barrier()

        if exc_type:
            logger.error(f"Skipping static kernel compilation due to the error: {exc_val}.")
            return

        if not self.is_global_first_rank:
            self.reselect_static_kernel()
            return

        all_compile = "+all" in self.static_kernels

        json_files_num = 0
        for rank_dir in self.cache_root.iterdir():
            if rank_dir.is_dir() and rank_dir.name.startswith("rank_"):
                debug_dirs = [d for d in rank_dir.iterdir() if d.is_dir() and d.name.endswith("_opcompile")]
                if not debug_dirs:
                    continue

                debug_dir = max(debug_dirs, key=lambda d: d.stat().st_mtime)
                rank_json_files = list(debug_dir.glob("*.json"))
                json_files_num += len(rank_json_files)

                for file in rank_json_files:
                    if not self.should_compile(all_compile, file.stem):
                        json_files_num -= 1
                        continue

                    if "Index" in file.stem or "FusedInferAttentionScore" in file.stem:
                        continue

                    copy_to_path = os.path.join(self.latest_cache_dir, f"{file.stem}_{rank_dir.name}{file.suffix}")
                    try:
                        shutil.copy2(file, copy_to_path)
                    except PermissionError:
                        logger.error(f"static kernel compilation copy file permissioin error")
                    except Exception as e:
                        logger.error(f"static kernel compilation copy file error, msg: {e.stderr}")

        if json_files_num <= 0:
            logger.info(f"No kernel files, skipping static compilation.")
            self.reselect_static_kernel()
            return

        compile_cpu_num = min(json_files_num, os.cpu_count() // 2)
        cmd = [
            "op_compiler",
            "-p", str(self.latest_cache_dir),
            "-v", torch_npu.npu.get_device_name(),
            "-l", "info",
            "-j", f"{compile_cpu_num}",
            "-o", str(self.cache_root),
        ]

        try:
            logger.info(f"Starting static kernel compilation process, kernels: {json_files_num}")
            start_time = time.time()
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            compile_time = time.time() - start_time
            logger.info("Static kernel compilation cost time: %.2f s", compile_time)
        except subprocess.CalledProcessError as e:
            logger.error(f"op_compiler execution failed, msg: {e.stderr}")
            self.reselect_static_kernel()
            return

        self.install_kernels()
        self.reselect_static_kernel()

    def reselect_static_kernel(self):
        if get_world_group().world_size > 1:
            get_world_group().barrier()
        torch_npu.npu._aclnn_reselect_static_kernel()
        if get_world_group().world_size > 1:
            get_world_group().barrier

    def install_kernels(self):
        if not self.is_global_first_rank:
            return

        for run_pkg in self.cache_root.glob("*.run"):
            filename = run_pkg.name
            try:
                result = subprocess.run([str(run_pkg)], check=True, capture_output=True, text=True)
                logger.info(f"Static kernel install successful")
            except subprocess.CalledProcessError as e:
                logger.error(f" {filename} install failed, msg: {e.stderr}")

    def should_compile(self, all_compile, stem):
        lower_stem = stem.lower()

        if not all_compile:
            for kernel in self.static_kernels:
                if lower_stem.startswith(kernel):
                    return True
            return False
        else:
            for kernel in self.disable_static_kernels:
                if lower_stem.startswith(kernel):
                    return False
            return True
    
    def uninstall(self):
        latest = Path(os.environ["ASCEND_HOME_PATH"])
        root = latest.parent
        pattern = f"*/opp/static_kernel/ai_core/uninstall.sh"
        matched = next(root.glob(pattern), None)
        if matched is not None:
            uninstall_path = str(matched)
            try:
                result = subprocess.run(
                    [uninstall_path], check=True, capture_output=True, text=True
                )
                logger.info(f"{uninstall_path} uninstall success")
            except subprocess.CalledProcessError as e:
                logger.error(f"{uninstall_path} uninstall failed, msg: \n{e.stderr}")
