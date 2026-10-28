# October 25 storage cleanup record

The authorized cleanup deleted 101 files (40,798,007,296 allocated bytes) and
pruned 2,247 unreachable Git objects (8,837,025,792 allocated bytes).
`cleanup_result.json` and `deletion_result.json` record execution and successful
verification. Removed preprocessing indexes and sorted intermediates must be
rebuilt before resuming workflows that require them.

The original full inventories and retained-file hash/identity lists were moved
to the ignored local archive on October 28. Cleanup decisions, exact deletion
records, prune lists and verification totals remain here. The
[evidence manifest](../../data/historical_evidence.json) identifies the four
archived files by path and hash; the [evidence index](../../data/README.md)
explains local access and recovery from the unchanged Git history.
