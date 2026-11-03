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
export PATH="$PWD/tools/codex/node_modules/.bin:$HOME/.local/bin:$PATH"

if [ -x "$PWD/data/antigravity-sdk/bin/python" ]; then
    export ROBOARCADE_AGY_PYTHON=${ROBOARCADE_AGY_PYTHON:-$PWD/data/antigravity-sdk/bin/python}
    export ROBOARCADE_ANTIGRAVITY_BACKEND=${ROBOARCADE_ANTIGRAVITY_BACKEND:-sdk}
fi

source "$PWD/.venv/bin/activate"
