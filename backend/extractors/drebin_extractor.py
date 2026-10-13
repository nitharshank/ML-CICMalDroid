"""
Drebin feature extractor using Androguard.

Maps an APK file to the 215-dim binary feature vector your ensemble was trained on.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Union

import numpy as np
from androguard.misc import AnalyzeAPK

log = logging.getLogger("malware-api")

SUSPICIOUS_API_PATTERNS = [
    "Ljava/lang/reflect/Method",
    "Ljava/lang/reflect/Field",
    "Ljava/lang/Runtime;->exec",
    "Ljava/lang/System;->loadLibrary",
    "Ldalvik/system/DexClassLoader",
    "Ldalvik/system/PathClassLoader",
    "Ljavax/crypto/Cipher",
    "Ljavax/crypto/spec/SecretKeySpec",
    "Landroid/telephony/TelephonyManager;->getDeviceId",
    "Landroid/telephony/TelephonyManager;->getSubscriberId",
    "Landroid/telephony/TelephonyManager;->getSimSerialNumber",
    "Landroid/telephony/SmsManager;->sendTextMessage",
    "Landroid/telephony/SmsManager;->sendMultipartTextMessage",
    "Landroid/content/pm/PackageManager;->getInstalledPackages",
    "Landroid/net/wifi/WifiManager;->getConnectionInfo",
    "Landroid/location/LocationManager;->getLastKnownLocation",
    "Ljava/net/URL;->openConnection",
    "Ljava/net/HttpURLConnection",
    "Lorg/apache/http/client/HttpClient",
    "Landroid/content/ContentResolver;->query",
    "Ljava/lang/Class;->forName",
    "Ljava/lang/ClassLoader",
    "Ljava/io/FileOutputStream",
]

_URL_RE = re.compile(rb"https?://[A-Za-z0-9._~:/\?#\[\]@!\$&'\(\)\*\+,;=%-]+")
_IP_RE = re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _norm(token: str) -> Set[str]:
    if not token:
        return set()
    if isinstance(token, bytes):
        token = token.decode("utf-8", errors="ignore")
    forms = {token}
    if "." in token:
        forms.add(token.rsplit(".", 1)[-1])
    if "/" in token:
        forms.add(token.rsplit("/", 1)[-1])
    return forms


def _any_in(tokens: Iterable[str], feature_set: Set[str]) -> Set[str]:
    matched = set()
    for t in tokens:
        for form in _norm(t):
            if form in feature_set:
                matched.add(form)
    return matched


class DrebinExtractor:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_set = set(feature_names)
        self.feat_idx = {f: i for i, f in enumerate(feature_names)}

    def extract(self, apk_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        apk_path = str(apk_path)
        log.info("=" * 60)
        log.info("[EXTRACTION] Starting APK analysis: %s", apk_path)
        log.info("=" * 60)

        log.info("[EXTRACTION] Step 1/6 — Parsing APK with Androguard...")
        a, d, dx = AnalyzeAPK(apk_path)
        log.info("[EXTRACTION] Package : %s", a.get_package())
        log.info("[EXTRACTION] App name: %s", a.get_app_name())
        log.info("[EXTRACTION] Version : %s", a.get_androidversion_name())
        log.info("[EXTRACTION] Min SDK : %s  |  Target SDK: %s", a.get_min_sdk_version(), a.get_target_sdk_version())

        tokens: Set[str] = set()

        log.info("[EXTRACTION] Step 2/6 — Extracting hardware features...")
        try:
            feats = a.get_features() or []
            tokens.update(feats)
            log.info("[EXTRACTION]   hardware features: %d", len(feats))
        except Exception as e:
            log.warning("[EXTRACTION]   get_features failed: %s", e)

        log.info("[EXTRACTION] Step 3/6 — Extracting permissions...")
        try:
            perms = a.get_permissions() or []
            tokens.update(perms)
            log.info("[EXTRACTION]   permissions declared: %d", len(perms))
            for p in perms:
                log.info("[EXTRACTION]     %s", p)
        except Exception as e:
            log.warning("[EXTRACTION]   get_permissions failed: %s", e)

        log.info("[EXTRACTION] Step 4/6 — Extracting components & intent filters...")
        try:
            activities = a.get_activities() or []
            services   = a.get_services()   or []
            receivers  = a.get_receivers()  or []
            providers  = a.get_providers()  or []
            tokens.update(activities)
            tokens.update(services)
            tokens.update(receivers)
            tokens.update(providers)
            log.info("[EXTRACTION]   activities: %d  services: %d  receivers: %d  providers: %d",
                     len(activities), len(services), len(receivers), len(providers))
        except Exception as e:
            log.warning("[EXTRACTION]   components failed: %s", e)

        try:
            for comp_type in ("activity", "service", "receiver"):
                comps = (activities if comp_type == "activity"
                         else services if comp_type == "service"
                         else receivers)
                for comp in comps:
                    try:
                        filters = a.get_intent_filters(comp_type, comp)
                        for action in filters.get("action", []):
                            tokens.add(action)
                            log.info("[EXTRACTION]     intent-action: %s", action)
                    except Exception:
                        pass
        except Exception as e:
            log.warning("[EXTRACTION]   intent_filters failed: %s", e)

        log.info("[EXTRACTION] Step 5/6 — Scanning DEX methods for suspicious APIs...")
        suspicious_hits = []
        try:
            method_count = 0
            for method in dx.get_methods():
                m = method.get_method()
                full = f"{m.get_class_name()}->{m.get_name()}"
                tokens.add(full)
                tokens.add(m.get_name())
                method_count += 1
                for pat in SUSPICIOUS_API_PATTERNS:
                    if pat in full:
                        suspicious_hits.append(pat)
                        tokens.add(pat)
                        log.info("[EXTRACTION]   [!] Suspicious API hit: %s  (in %s)", pat, full)
            log.info("[EXTRACTION]   methods scanned: %d  suspicious hits: %d", method_count, len(suspicious_hits))
        except Exception as e:
            log.warning("[EXTRACTION]   dex method scan failed: %s", e)

        log.info("[EXTRACTION] Step 6/6 — Scanning strings for URLs and IPs...")
        urls, ips = self._scan_strings(d)
        tokens.update(urls)
        tokens.update(ips)
        log.info("[EXTRACTION]   URLs found: %d  IPs found: %d", len(urls), len(ips))

        vec = np.zeros(len(self.feature_names), dtype=np.float32)
        matched = _any_in(tokens, self.feature_set)
        for f in matched:
            vec[self.feat_idx[f]] = 1.0

        log.info("[EXTRACTION] Feature vector: %d/%d features active", int(vec.sum()), len(self.feature_names))
        log.info("=" * 60)

        meta = {
            "package_name": a.get_package(),
            "app_name": str(a.get_app_name() or ""),
            "version_name": a.get_androidversion_name(),
            "min_sdk": a.get_min_sdk_version(),
            "target_sdk": a.get_target_sdk_version(),
            "permissions_declared": len(a.get_permissions() or []),
            "activities": len(activities),
            "services": len(services),
            "receivers": len(receivers),
            "providers": len(providers),
            "suspicious_api_hits": sorted(set(suspicious_hits)),
            "urls_found": len(urls),
            "ips_found": len(ips),
            "matched_features_count": int(vec.sum()),
            "matched_features": sorted(matched),
        }
        return vec, meta

    @staticmethod
    def _scan_strings(d_list) -> Tuple[Set[str], Set[str]]:
        urls, ips = set(), set()
        if not d_list:
            return urls, ips
        if not isinstance(d_list, list):
            d_list = [d_list]
        for d in d_list:
            try:
                for s in d.get_strings():
                    raw = s.encode("utf-8", errors="ignore") if isinstance(s, str) else s
                    for m in _URL_RE.findall(raw):
                        urls.add(m.decode("utf-8", errors="ignore"))
                    for m in _IP_RE.findall(raw):
                        ips.add(m.decode("utf-8", errors="ignore"))
            except Exception as e:
                log.warning("string scan failed: %s", e)
        return urls, ips