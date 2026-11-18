# Two-collector protocol validation

`revision_protocol_validation.py` validates and times the actual exchange of
round contracts, per-client field shares, acknowledgements and aggregate shares
over length-prefixed JSON/TCP. Every completed round is checked against the
plaintext sum of the same clipped and quantized synthetic updates. The collector
retains a running share sum and roster, and the reports contain no individual
updates or shares.

This is a **protocol microbenchmark with synthetic updates**, not end-to-end
detector training. All client update generation and orchestration remain in one
trusted process. The two collectors must be honest-but-curious and noncolluding.
The module does not provide TLS, authentication, differential privacy, malicious
participant protection, or dropout recovery. An incomplete or rejected round
does not yield an accepted reconstructed aggregate. This is not a distributed
transaction/commit protocol: one collector may already have returned its sum
when the other connection fails, but the orchestrator does not accept a result.

The separate existing `SecureAggregationSession` still implements the training
pipeline's local subprocess/IPC transport. The TCP harness exercises the same
clipping, quantization, field and sharing primitives, but it is an additional
benchmark transport, not a replacement silently applied to training.

## Same-host check

The protocol regression suite exercises the same-host benchmark, whose library
function starts two bounded collector subprocesses on `127.0.0.1` using
automatically assigned ports. Each roster size uses a fresh connection/session.
The output directory must be new; a failure writes `complete: false` and the
error to `protocol_summary.json`. It never marks partial measurements complete.

`protocol_rounds.csv` records elapsed time, quantization/sharing time, framed
JSON upload/download byte counts, message count and plaintext-equivalence checks.
`protocol_summary.json` reports median/p95 elapsed time for each roster size.
Byte counts include JSON and the four-byte length prefix, but exclude TCP/IP
headers, retransmissions and operating-system transport overhead. These are
application bytes actually sent/read, not a packet-capture measurement. Logical
uint64 field-element payload is reported separately. Round timing excludes
collector startup and connection establishment; no detector training is timed.

All default results say `transport_environment: same_host_loopback` and
`independent_physical_hosts_verified: false`. Do not label them as multi-host
validation.

## User-controlled LAN endpoints

For a full review run, copy `external/lan_manifest.example.json` to
`external/lan_manifest.json` and record the actual orchestrator, two collector
hosts and endpoints. Start the two collectors using the commands below with
`--sessions 6` for the two head dimensions (1,035 and 690), then run
`main.py` on the orchestrator with the benchmark data ready. Each dimension measures 30 rounds at 3, 5 and 10
logical clients. `--lan-manifest` selects another declaration, while
`--preflight-only` checks its format without connecting. Physical-host identity
is operator-declared; the program verifies protocol responses but cannot
independently attest the hardware. Client update generation remains on the
orchestrator and the run measures synthetic-update transport.
The fresh workflow runs LAN measurement before training. Start the collectors
shortly before clicking the experiment entry.

Only expose the collectors on a trusted, controlled test network. The address
must be explicitly supplied to listen remotely; the default bind is loopback.
Run each collector on the intended machine, substituting its assigned LAN IP:

```text
python main.py --collector --bind COLLECTOR_A_LAN_IP --port 8761 --sessions 6 --timeout 60
python main.py --collector --bind COLLECTOR_B_LAN_IP --port 8762 --sessions 6 --timeout 60
```

On the benchmark orchestrator, click **FedCAID 一键运行** with the LAN manifest in place.

`--sessions` must cover the number of distinct roster-size conditions. A
collector serves at most that many TCP sessions and exits. It also exits after
an idle accept timeout of at least 30 seconds; start the orchestrator promptly
or increase `--timeout`. A protocol failure closes that session, and a benchmark
failure stops the run rather than accepting a smaller roster.

The receipt labels explicitly supplied addresses as `user_supplied_endpoints`;
different IP addresses alone do not prove independent machines or trust domains.
For a multi-host paper result, separately record physical host identities, OS,
CPU/RAM, network topology, configured delay/bandwidth/loss, and which processes
run on each host. Packet-capture bytes, if measured, need a separate field and
must not be confused with the application-byte counter. No multi-host result is
claimed by the implementation or the default local test.

## Regression tests

```text
python -m unittest discover -s tests -p test_revision_protocol_validation.py -v
```

The tests check real local TCP round equivalence, incomplete-roster failure,
bounded framing, duplicate/stale/invalid shares, and existing local-IPC failure
contracts including worker exit and deterministic timeout handling. Existing
`test_revision_adaptation.py` supplies exhaustive toy-field single-share
uniformity and missing-client failure tests. Together these provide executable
evidence for the stated construction and failure behavior; they are not a
cryptographic audit or proof of malicious-party security.
