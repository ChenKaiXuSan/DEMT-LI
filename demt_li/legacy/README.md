# Original network implementation

These files are kept byte-for-byte as used in the reported experiments so that the
released checkpoints load with `strict=True` and reproduce the saved predictions.

| Method | Model file |
| --- | --- |
| `image_2d` (DEMT-LI, 2D coronal image surrogate) | `image_2d/Time_series_seq2seq_4DCT_voxelmorph.py` |
| `rpm_1d` (1D diaphragm-position surrogate) | `rpm_1d/Time_series_seq2seq_4DCT_voxelmorph.py` |
| `no_surrogate` | `no_surrogate/seq2seq_4DCT_voxelmorph.py` |

`common/` holds the shared ConvLSTM cell, the VoxelMorph-style `SpatialTransformer`
(`grid_sample` with the PyTorch default `align_corners=False`) and `ConvOut`.
The models are loaded through `demt_li.network.build_model`; do not import these
files directly.

The architecture follows RMSim (Lee et al., Phys. Med. Biol. 68:045009, 2023) and
the spatial transformer follows VoxelMorph (Balakrishnan et al., IEEE TMI 2019).
