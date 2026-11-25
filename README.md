# FedVista

**FedVista: Robust Federated VLA Learning from Diverse Visual Worlds**

Anonymous research code and project website.

- [Project website](https://anonymous-for-sub.github.io/FedWorldMix/)
- [Paper](https://anonymous-for-sub.github.io/FedWorldMix/assets/papers/FedVista.pdf)
- [Supplementary material](https://anonymous-for-sub.github.io/FedWorldMix/assets/papers/FedVista_supplement.pdf)
- [Code and setup](code/README.md)

`code/` contains the existing VistaMix implementation, training configurations, backbone integrations and CPU tests. `docs/` contains the static GitHub Pages website and local media assets.

```bash
pip install -e 'code/.[test]'
python -m pytest code/tests -q
```

The website is served from `main:/docs`. To preview locally, run `python -m http.server 8000 --directory docs` and open `http://localhost:8000`.

The website adapts the Long-WAM page under Apache 2.0; see [website attribution](docs/NOTICE.md). Existing HPT third-party notices are preserved in [code attribution](code/THIRD_PARTY.md).
