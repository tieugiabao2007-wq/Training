from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import SETTINGS
from gold_ai.data.providers import get_provider
from gold_ai.data.quality import validate_and_clean_bars


provider = get_provider(SETTINGS)
raw = provider.fetch_bars()
clean, report = validate_and_clean_bars(raw)
stem = f"{provider.metadata.instrument.replace('=', '_')}_{provider.metadata.interval}"
paths = provider.save(clean, stem)
print({"data": str(paths[0]), "metadata": str(paths[1]), "quality": report.to_dict()})

