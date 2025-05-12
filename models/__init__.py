from .BlindShadowRemoval import BlindShadowRemoval
from .DeShadowNet import DeShadowNet
from .DHAN import DHAN
from .DMTN import DMTN
from .FSRNet import FSRNet
from .PSM import PSM
from .Homoformer import HomoFormer
from .Lyu import Lyu
from .TBRNet import TBRNet
from .CIRNet import CIRNet
from .DC_ShadowNet import DC_ShadowNet
from .BMNet import BMNet
from .S3R_Net import S3R_Net
from .Shadow_R import Shadow_R
from .ST_CGAN import ST_CGAN
from .MaskShadowGAN import MaskShadowGAN
from .G2R_ShadowNet import G2R_ShadowNet
from .Unfolding import Unfolding
from .ShadowFormer import ShadowFormer
from .DSC import DSC
from .RASM import RASM
from .SADC import SADC
from .PRNet import PRNet
from .SG_ShadowNet import SG_ShadowNet
from .LG_ShadowNet import LG_ShadowNet
from .model import Model


model_registry = {
    'BlindShadowRemoval': BlindShadowRemoval,
    'DeShadowNet': DeShadowNet,
    'DHAN': DHAN,
    'DMTN': DMTN,
    'FSRNet': FSRNet,
    'PSM': PSM,
    'HomoFormer': HomoFormer,
    'Lyu': Lyu,
    'TBRNet': TBRNet,
    'CIRNet': CIRNet,
    'DC_ShadowNet': DC_ShadowNet,
    'BMNet': BMNet,
    'S3R_Net': S3R_Net,
    'Shadow_R': Shadow_R,
    'ST_CGAN': ST_CGAN,
    'MaskShadowGAN': MaskShadowGAN,
    'G2R_ShadowNet': G2R_ShadowNet,
    'Unfolding': Unfolding,
    'ShadowFormer': ShadowFormer,
    'DSC': DSC,
    'RASM': RASM,
    'SADC': SADC,
    'PRNet': PRNet,
    'SG_ShadowNet': SG_ShadowNet,
    'LG_ShadowNet': LG_ShadowNet,
    'Model': Model
}