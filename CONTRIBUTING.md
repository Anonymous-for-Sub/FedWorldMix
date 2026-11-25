# Contributing

For a bug report, include the command, configuration, Python version, dependency versions and relevant error output. A small reproducible example is useful for gradient-composition and aggregation issues.

For a change, describe the behavior it affects and run:

```bash
python -m pytest code/tests -q
python scripts/check_site.py
node --check docs/assets/js/site.js
node --check docs/assets/js/overview.js
```

HPT changes should also pass `python -m pytest code/hpt_tests -q` in an environment with PyTorch. Update the relevant documentation when configuration fields or public behavior change. Preserve existing third-party notices.

Pull requests are welcome under the repository's Apache-2.0 software license. Keep paper values, checkpoints and experiment settings unchanged unless the change explicitly concerns them.
