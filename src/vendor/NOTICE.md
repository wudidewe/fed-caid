# Third-party provenance

These vendored files retain the numerical implementations used in the validated
experiment. The SHA256 values below identify the copies in this repository.

| Module | Embedded version | SHA-256 |
| --- | --- | --- |
| tabm.py | 0.0.3 | 5c49b08847961c6110c9d7c5dda2a017de6431e3502840ebb910feca1e3fd2f8 |
| rtdl_num_embeddings.py | 0.0.12 | 48dfc019d447e0cd225deb8b8145b3efa8b42cde9aa314772d6e234753e1cee1 |

Original sources:

- https://github.com/yandex-research/tabm
- https://github.com/yandex-research/rtdl-num-embeddings

TabM retains its Apache-2.0 license in `TABM_LICENSE.txt`.
The numerical embedding module retains its MIT license in `RTDL_LICENSE.txt`.
This repository changed the TabM import of `rtdl_num_embeddings` to a relative
import of the bundled module. The validated copy before that import adjustment
had SHA256 `fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade`.
The numerical implementations and licence text match the validated copies.
The experimental application code is separate from these components.
