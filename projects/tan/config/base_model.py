from projects.tan.modeling import TAN, FrameAvgPool, BaseFusion, MapConv, \
    TextEncoderGlove, build_criterion, SparsePropMaxPool, SparsePropConv
from kn_util.config import LazyCall as L

model = L(TAN)(video_encoder=L(FrameAvgPool)(input_size=1024, hidden_size=512, kernel_size=4, stride=4),
               text_encoder=L(TextEncoderGlove)(global_glove_key="glove", freeze=True),
               fusion_module=L(BaseFusion)(hidden_size=512,
                                           txt_input_size=300,
                                           txt_hidden_size=512,
                                           num_layers=3,
                                           bidirecitonal=False),
               map_conv=L(MapConv)(input_size=512,
                                   hidden_sizes=[512] * 4,
                                   kernel_sizes=[9] * 4,
                                   strides=[1] * 4,
                                   paddings=[16] + [0] * 3,
                                   dilations=[1] * 4),
               prop_module=L(SparsePropConv)(hidden_size=512, num_scale_layers=[16, 8, 8]),
               criterion=L(build_criterion)(min_iou=0.5, max_iou=1.0, bias=0.0),
               pred_input_size=512,
               iou_threshold=0.5)
