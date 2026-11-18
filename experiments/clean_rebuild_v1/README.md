# Clean Rebuild v1

This experiment preserves all historical D05 results and creates a new, one-way
development protocol. AWID3 Evil Twin and D01-D03 are development data. D05 is
never available to training, calibration, feature/hyperparameter/model selection,
or threshold selection.

Dataset roles are defined in `protocol.yaml`. The firewall is enforced by
`scripts/clean_experiment.py` and tested in `tests/test_clean_experiment.py`.

## Lifecycle

1. Validate the protocol:
   `.venv/bin/python scripts/clean_experiment.py validate`
2. Develop a new model without reading D05:
   `.venv/bin/python scripts/run_clean_rebuild_v1.py develop`
   Artifacts are written under `models/clean_rebuild_v1/`; the historical
   `models/selected_model.joblib` is never reused as the clean artifact.
3. Freeze hashes:
   `.venv/bin/python scripts/clean_experiment.py freeze`
4. Verify the immutable-style lock before any D05 access:
   `.venv/bin/python scripts/clean_experiment.py authorize-d05`
5. Evaluate D05 once:
   `.venv/bin/python scripts/run_clean_rebuild_v1.py evaluate-d05`
   Results go to a new directory and are reported as a **Previously Inspected
   External Test**, not a pristine independent test.

Changing any required input after step 3 invalidates authorization. Create a new
experiment version and freeze again rather than overwriting this experiment.
