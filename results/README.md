# Result files

The dated `comparison_YYYYMMDD.csv` files capture earlier iterations of the
research. They use the accounting, sizing and pivot definitions that were in the
project at the time, so they are useful as history but should not be compared
directly with the current output.

Files named `comparison_validated_YYYYMMDD.csv` come from engine version 0.2.
That version uses daily marked-to-market equity, contract-aware futures sizing,
complete trade logs and fixed-lag pivot confirmation. Each row includes the
sample dates, modeled trading costs, data-quality counts and a SHA-256
fingerprint of the downloaded price frame.

Yahoo can revise continuous-futures histories. Keep the generated CSV alongside
any quoted result so the exact run remains identifiable.
