#!/usr/bin/env python3
"""
Alias for run_sam3_som_phase.py - for backward compatibility
"""
import sys
from pathlib import Path

script_dir = Path(__file__).parent
target_script = script_dir / "run_sam3_som_phase.py"

# Replace this script with the target one
sys.argv[0] = str(target_script)

# Execute the target script
with open(target_script, 'rb') as f:
    code = compile(f.read(), str(target_script), 'exec')
    exec(code, globals(), locals())
