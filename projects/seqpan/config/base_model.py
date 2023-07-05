from projects.seqpan.modeling.layers import (Embedding, FeatureEncoder, VisualProjection, DualAttentionBlock, CQAttention, CQConcatenate,
                     Conv1D, SeqPANPredictor)
from projects.seqpan.modeling.arch import SeqPAN
from kn_util.basic import global_get

d_model = "${model_cfg.d_model}"
dropout = "${model_cfg.dropout}"
d_word = "${model_cfg.d_word}"
d_char = "${model_cfg.d_char}"
SeqPAN(text_encoder=Embedding(num_words=num_words, num_chars=num_chars, out_dim=d_model,
                                       word_dim=word_dim, 
                                       char_dim=char_dim,
                                       word_vectors=L(global_get)(name="word_vectors"),
                                       droprate=dropout)