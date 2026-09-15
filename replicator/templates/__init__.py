"""Family templates. A template is a list of nodes `(inputs, config) -> outputs` with the
paper-specific nodes left for the builder to fill. Node outputs are cached by joblib on the
hash of inputs+config, so one config flip re-runs only what depends on it."""
