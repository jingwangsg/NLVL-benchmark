from data.datapipe import VideoPipeHDF5, TextPipeGlove, TextPipePretrained
from kn_util.config import LazyCall as L

video_hdf5 = L(VideoPipeHDF5)(hdf5_file="${paths.data}/${data.dataset}/${data.feat_type}.hdf5",
                              seq_len="${data.video_length}")

text_glove = L(TextPipeGlove)(glove="${data.glove}", cache="${paths.data}/.cache")
