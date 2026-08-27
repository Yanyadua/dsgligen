# Generation Quality Degradation Attribution Plan

1. Add failing unit tests for fuser restoration and graph-gate override.
2. Add evaluation-only controls and metadata fields.
3. Run unit tests and syntax checks locally/remote.
4. Push the evaluator change to the remote recovery tree.
5. Sample F0G0, F0G1, F1G0, and F1G1 on fixed indices.
6. Verify protocol metadata and image counts.
7. Build and inspect a labeled comparison grid.
8. Commit the evaluator and tests for rollback.
