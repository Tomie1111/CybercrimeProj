# Phase 4.2 Benchmark Methodology

- Primary measurement: detection-only on one pre-extracted D05-A Beacon/W20 input artifact.
- W20 membership hashes must exactly equal Phase 4.1 D05-A membership.
- Warm-up and repetitions are fixed in `configs/phase4_2_lightweight_benchmark.yaml`.
- Each method/unit executes in a new child PID. Raw per-unit wall/CPU samples are retained.
- Per-Beacon and Per-W20 results are separate; `STATIC` and `RF` have no Beacon result.
- Pipeline reference uses one instrumented PCAP pass. Repeated detection does not decode PCAP.
- Scalability replication represents compute load only; it has no new labels or detection metrics.
