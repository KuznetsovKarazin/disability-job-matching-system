# Blockchain Anchoring Benchmark Results
Generated: 2025-08-22T21:13:24.834678

## Environment
- Python: 3.13.5
- CPU: 16 cores @ 2516 MHz
- RAM: 62.8 GB total
- Platform: win32

## Build Performance
- 100 records: 2.15s avg, 0.1MB peak
- 1,000 records: 25.50s avg, 0.8MB peak
- 10,000 records: 236.83s avg, 22.1MB peak

## Proof Generation Performance
### O(log n) Performance (with saved levels)
- 100 records: 1.61ms avg proof time
- 1,000 records: 3.76ms avg proof time
- 10,000 records: 25.33ms avg proof time

## Verification Performance
- Average verification time: 0.90ms
- Success rate (correct password): 0.00%
- False positive rate (wrong password): 0.00%

## Key Insights
- Memory scaling: 222.8x increase for 100.0x more records
- KDF iterations impact:
  - 10,000 iterations: 11.21s avg
  - 50,000 iterations: 63.46s avg
  - 100,000 iterations: 108.94s avg
- Proof length scales with log(n):
  - 100 records: 7.0 nodes (expected ~7)
  - 1,000 records: 10.0 nodes (expected ~10)
  - 10,000 records: 14.0 nodes (expected ~14)

## Files Generated
- build_metrics.csv: Build phase performance data
- proof_metrics.csv: Proof generation statistics
- verify_metrics.csv: Verification timing and correctness
- anchor_metrics.csv: Blockchain interaction performance
- environment_info.json: System configuration
- bench.log: Detailed execution log
- performance_charts.png: Key metrics visualization