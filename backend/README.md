# Android Malware Detection — Backend

FastAPI service that accepts an APK upload, runs Drebin-style feature
extraction with Androguard, and classifies it through your trained ensemble
(RF + XGBoost + MLP).

## 1. Project layout (after you drop in artefacts)

```
backend/
├── main.py
├── requirements.txt
├── extractors/
│   └── drebin_extractor.py
├── utils/
│   └── model_loader.py
├── processed/                       <-- copy from your training output
│   ├── drebin_features.json
│   ├── drebin_scaler.joblib
│   └── drebin_variance_selector.joblib
└── models/                          <-- copy from your training output
    ├── rf_drebin.joblib
    ├── xgb_drebin.joblib
    └── mlp_drebin.joblib
```

## 2. Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on Androguard**: 3.4.0a1 works on Python 3.9–3.11. If you're on 3.12+,
> install from git: `pip install git+https://github.com/androguard/androguard.git`.

## 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit `http://localhost:8000/docs` for interactive API docs.

## 4. Test from the command line

```bash
# health check
curl http://localhost:8000/health

# predict
curl -X POST -F "file=@path/to/sample.apk" http://localhost:8000/predict
```

## 5. Connecting the Android app

| Where the app runs              | Backend URL to use                              |
|---------------------------------|-------------------------------------------------|
| Android emulator (same laptop)  | `http://10.0.2.2:8000/`                         |
| Physical phone on same Wi-Fi    | `http://<your-laptop-LAN-IP>:8000/`             |

To find your LAN IP:
- macOS / Linux: `ifconfig | grep inet` or `ip addr`
- Windows:       `ipconfig`

If the phone can't reach the laptop, your laptop firewall is almost certainly
the cause — allow inbound TCP 8000.

## 6. Notes on accuracy

The extractor scrapes manifest metadata, DEX method names, intent filters, and
DEX strings, then matches everything against the **exact feature names from
your training JSON**. If your training JSON uses short forms like `SEND_SMS`
the extractor handles that; if it uses fully-qualified forms like
`android.permission.SEND_SMS`, that also works.

If you find a known-malicious APK is misclassified, run:

```bash
curl -X POST -F "file=@suspect.apk" http://localhost:8000/predict | jq .extracted.matched_features
```

…to see which of the 215 training features were detected. Sparse matches mean
the manifest/DEX uses naming conventions different from your training data,
and the `_norm()` function in `drebin_extractor.py` may need extending.
