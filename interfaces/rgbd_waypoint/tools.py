"""The tool list the MCP server loads: implementation.py's tools."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('rgbd_tools',Path(__file__).with_name('implementation.py'))
_impl=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
TOOLS=_impl.TOOLS
