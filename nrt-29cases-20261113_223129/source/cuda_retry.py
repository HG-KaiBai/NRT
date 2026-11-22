"""Run a module and return Slurm's retry status for CUDA availability failures.

Keep launch/retry policy outside the scientific modules: completed results and
running workers must retain their existing source fingerprints.
"""
import runpy
import sys
import traceback


RETRY_EXIT_CODE = 75
CUDA_UNAVAILABLE = (
    "cuda-capable device(s) is/are busy or unavailable",
    "all cuda-capable devices are busy or unavailable",
    "cudaerrordevicesunavailable",
    "no cuda gpus are available",
    "found no nvidia driver on your system",
    "cuda driver initialization failed",
    "cuda error: initialization error",
    "cuda error: system not yet initialized",
    "cuda driver version is insufficient for cuda runtime version",
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Usage: python -m cuda_retry MODULE [ARGS ...]")
    module, *arguments = args
    previous = sys.argv
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except Exception as error:
        if not any(message in str(error).lower() for message in CUDA_UNAVAILABLE):
            raise
        traceback.print_exc()
        print("CUDA device unavailable; exiting 75 so runtime.sh can requeue this task.",
              file=sys.stderr, flush=True)
        raise SystemExit(RETRY_EXIT_CODE) from error
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
