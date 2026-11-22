# Changelog

## Unreleased — repository finalization (2026-11-22)

- Replaced the initial-setup README with the audited project overview, dataset inventory, fixed split/manifest protocol, model descriptions, results, limitations, environment, and read-only verification commands.
- Synchronized `requirements.txt` to the captured 57-package macOS Apple-Silicon environment (Python 3.11.4); explicitly documented that `tensorflow-metal` makes this environment platform-specific.
- Documented the saved-prediction results workflow and checkpoint hash/load audit workflow.
- Documented dataset/source-confounding and repeated-Test-evaluation limitations.
- Retained the historical v1.0 changelog entry below.

## v1.0 — 2026-11-19

- Added a shared resumable trainer for 10%, 25%, 50%, and 100% frozen manifests.
- Added manifest/source SHA256, patient-group completeness, class count, and Train/Validation separation checks.
- Disabled TensorFlow Meta Optimizer before model construction and tracing.
- Added one-epoch-per-invocation training sessions with BackupAndRestore.
- Selected best checkpoint using scikit-learn AUROC computed from `model.predict()` on Validation.
- Corrected validation preprocessing contract: `prepare_batch` is applied once when the dataset is built; prediction callback never normalizes again.
- Added read-only checkpoint/prediction verification script.
- Test is not loaded or evaluated by the trainer.
