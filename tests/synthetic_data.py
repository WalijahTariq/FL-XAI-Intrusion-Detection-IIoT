"""Tiny synthetic dataset with the exact 63-column schema of ML-EdgeIIoT-dataset.csv.

Used ONLY by the automated tests so CI can check that the code runs end to end
in seconds. The numbers are random and mean nothing; real results always come
from the real dataset (scripts/run_pipeline.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns removed by preprocessing (identifiers / free text) -- see src/config.py.
TEXT_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
    "http.file_data", "http.request.uri.query", "http.request.full_uri", "http.referer",
    "tcp.options", "tcp.payload", "tcp.srcport", "dns.qry.name", "mqtt.msg", "mqtt.topic",
    "mqtt.msg_decoded_as",
]
CATEGORICAL_COLUMNS = {
    "http.request.method": ["0", "GET", "POST"],
    "http.request.version": ["0", "HTTP/1.1", "HTTP/1.0"],
    "mqtt.protoname": ["0", "MQTT"],
}
NUMERIC_COLUMNS = [
    "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp",
    "icmp.unused", "http.content_length", "http.response", "http.tls_port", "tcp.ack",
    "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst",
    "tcp.connection.syn", "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack",
    "tcp.len", "tcp.seq", "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name.len",
    "dns.qry.qu", "dns.qry.type", "dns.retransmission", "dns.retransmit_request",
    "dns.retransmit_request_in", "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype", "mqtt.proto_len", "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]
# The 15 raw classes of the ML file (MITM/Fingerprinting are dropped, Ransomware -> Zero_Day).
RAW_CLASSES = [
    "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP", "Fingerprinting", "MITM",
    "Normal", "Password", "Port_Scanning", "Ransomware", "SQL_injection", "Uploading",
    "Vulnerability_scanner", "XSS",
]


def make_synthetic_frame(seed: int = 0, min_rows: int = 25, max_rows: int = 70) -> pd.DataFrame:
    """Return an imbalanced synthetic frame (so SMOTE has something to balance)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, cls in enumerate(RAW_CLASSES):
        n = int(rng.integers(min_rows, max_rows))
        data = {col: rng.normal(loc=i * 0.5, scale=1.0, size=n).round(4) for col in NUMERIC_COLUMNS}
        for col, choices in CATEGORICAL_COLUMNS.items():
            data[col] = rng.choice(choices, size=n)
        for col in TEXT_COLUMNS:
            data[col] = [f"{col}-{cls}-{k}" for k in range(n)]
        data["Attack_label"] = 0 if cls == "Normal" else 1
        data["Attack_type"] = cls
        parts.append(pd.DataFrame(data))
    frame = pd.concat(parts, ignore_index=True)
    columns = TEXT_COLUMNS + list(CATEGORICAL_COLUMNS) + NUMERIC_COLUMNS + ["Attack_label", "Attack_type"]
    assert len(columns) == 63
    return frame[columns].sample(frac=1.0, random_state=seed).reset_index(drop=True)
