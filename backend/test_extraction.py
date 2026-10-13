"""
Standalone test for the feature extraction + prediction pipeline.

Use this to verify the backend works on a known APK before connecting
the Android app.

Usage:
    python test_extraction.py path/to/sample.apk
"""
import json
import sys
from pathlib import Path

from extractors.drebin_extractor import DrebinExtractor
from utils.model_loader import EnsembleModel


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_extraction.py <path-to-apk>")
        sys.exit(1)

    apk_path = sys.argv[1]
    if not Path(apk_path).exists():
        print(f"File not found: {apk_path}")
        sys.exit(1)

    base_dir = Path(__file__).parent.parent.resolve()

    print("Loading models…")
    ensemble = EnsembleModel(base_dir)
    ensemble.load()

    print("Extracting features…")
    extractor = DrebinExtractor(ensemble.features)
    vec, meta = extractor.extract(apk_path)

    print("Running ensemble…")
    probs = ensemble.predict(vec)

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"Package:           {meta['package_name']}")
    print(f"App name:          {meta['app_name']}")
    print(f"Active features:   {int(vec.sum())} / {len(ensemble.features)}")
   # print(f"Suspicious APIs:   {len(meta['suspicious_api_hits'])}")
    print()
   # print(f"Random Forest:     {probs['rf']*100:6.2f}%")
   # print(f"XGBoost:           {probs['xgb']*100:6.2f}%")
   # print(f"MLP:               {probs['mlp']*100:6.2f}%")
    print(f"Ensemble:          {probs['ensemble']*100:6.2f}%")
    print()
    verdict = "MALWARE" if probs["ensemble"] >= 0.5 else "BENIGN"
    print(f"VERDICT:           {verdict}")
    print("=" * 50)

    if int(vec.sum()) < 5:
        print("\n⚠️  Few features matched. Either the APK is genuinely "
              "minimal, or the feature naming in your training JSON differs "
              "from what Androguard produces. Inspect `meta['matched_features']` "
              "and consider extending `_norm()` in drebin_extractor.py.")
        print("\nFirst 20 raw tokens seen:")
        print(json.dumps(meta["matched_features"][:20], indent=2))


if __name__ == "__main__":
    main()
