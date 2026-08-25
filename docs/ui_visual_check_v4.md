# UI Visual Check — BioSignal-FM V4

**Environment:** Local Streamlit at `127.0.0.1:8501`  
**Historical status:** Passed

The V4 home page loaded and displayed the platform identity, page sequence, EMG/ECG/EEG/ECoG/iEEG/fNIRS status, and an explicit message that synthetic data is not benchmark or clinical evidence. The first inspection found clipping in two long-value summary cards. The experimental-path value was shortened to `ECoG` and the compact evidence-status value to `on`, while surrounding copy retained the full context. The final inspection confirmed that all four summary cards were fully readable at the standard desktop viewport.

| Check item | Result |
|---|---|
| Streamlit application load | Passed |
| V4 identity and modality display | Passed |
| Evidence-boundary message | Passed |
| Summary-card readability | Passed after responsive-value adjustment |
| Visual hierarchy and color contrast | Clear and consistent on the tested light background |

This record is a limited local visual smoke check. It is not a substitute for an accessibility audit, cross-browser/device testing, usability research, or validation of scientific content.
