import sys
from pathlib import Path

# Add smart-farm directory to python path for pytest execution
smart_farm_dir = Path(__file__).parent.parent / "smart-farm"
if str(smart_farm_dir) not in sys.path:
    sys.path.insert(0, str(smart_farm_dir))
