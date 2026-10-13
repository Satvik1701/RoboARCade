export PYTHONPATH=$PWD

# the point-cloud deproject() is thin memory-bound matmuls: slower multi-threaded
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# headless rendering; macOS has no EGL
case "$(uname -s)" in
    Darwin) _mujoco_gl_default=cgl ;;
    *)      _mujoco_gl_default=egl ;;
esac
export MUJOCO_GL=${MUJOCO_GL:-$_mujoco_gl_default}
unset _mujoco_gl_default

export ROBOARCADE_ASSETS_DIR=${ROBOARCADE_ASSETS_DIR:-$PWD/assets}

source "$PWD/.venv/bin/activate"
