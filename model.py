from module import ( double_conv, out_conv, uconv, max_pool3d, cat, 
                     res2_block, res1_block, res1_block_2, scaling_target,
                     trans_conv, res2_se_block, res2_block_nl, res2_block_2,
                     res2_cm_block, res2_cm_block_v2
                   )
from itertools import product
from einops import rearrange

import torch.nn.functional as F
import torch.nn as nn
import module as MD
import numpy as np
import torch



##=============================================================================
class LD_UNet( nn.Module ):
    def __init__( self, in_channel, 
                        out_channel, 
                        base_channels = 32, 
                        max_channels  = 320 
                 ):
        super( LD_UNet, self ).__init__()
        ##
        ic = in_channel
        bc = base_channels
        mc = max_channels
        
        ## Encoder
        self.block_0_0 = res2_block( ic  , bc  , [1,3,3] ) # 256
        self.block_1_0 = res2_block( bc  , bc*2, [1,3,3] ) # 128
        self.block_2_0 = res2_block( bc*2, bc*4, [1,3,3] ) # 64
        self.block_3_0 = res2_block( bc*4, bc*8, [1,3,3] ) # 32
        self.block_4_0 = res2_cm_block_v2( bc*8, mc  , [1,3,3], [1,16,16] ) # 16
        self.block_5_0 = res2_cm_block_v2( mc  , mc  , [1,3,3], [1,8,8] ) # 8
        self.block_6_0 = res2_cm_block_v2( mc  , mc  , [1,3,3], [1,4,4] ) # 4
  
        ## Decoder
        self.block_0_1 = MD.res2_block_4( bc*2    , bc  , [1,3,3], [1,13,13], groups_2=16 )
        self.block_1_1 = MD.res2_block_4( bc*4    , bc*2, [1,3,3], [1,11,11], groups_2= 8 )
        self.block_2_1 = MD.res2_block_4( bc*8    , bc*4, [1,3,3], [1, 9, 9], groups_2= 4 )
        self.block_3_1 = MD.res2_block_4( bc*16   , bc*8, [1,3,3], [1, 7, 7], groups_2= 2 )
        self.block_4_1 = res2_cm_block_v2( mc+mc   , mc  , [1,3,3], [1,16,16] )
        self.block_5_1 = res2_cm_block_v2( mc+mc   , mc  , [1,3,3], [1, 8, 8] )
        
        
        ## Up_conv
        self.up_1 = uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = uconv( mc  , bc*8, [1,2,2] )           
        self.up_5 = uconv( mc  , mc  , [1,2,2] )           
        self.up_6 = uconv( mc  , mc  , [1,2,2] )           
       
        ## Out_conv
        self.out_0 = out_conv( bc  , out_channel )
        self.out_1 = out_conv( bc*2, out_channel )
        self.out_2 = out_conv( bc*4, out_channel )
        self.out_3 = out_conv( bc*8, out_channel )
        self.out_4 = out_conv( mc  , out_channel )
        self.out_5 = out_conv( mc  , out_channel )

        
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( max_pool3d( x_3_0, [1, 2, 2] ) )
        x_5_0 = self.block_5_0( max_pool3d( x_4_0, [1, 2, 2] ) )
        x_6_0 = self.block_6_0( max_pool3d( x_5_0, [1, 2, 2] ) )

        x_5_1 = self.block_5_1( cat( self.up_6( x_6_0 ), x_5_0 ) )        
        x_4_1 = self.block_4_1( cat( self.up_5( x_5_1 ), x_4_0 ) )
        x_3_1 = self.block_3_1( cat( self.up_4( x_4_1 ), x_3_0 ) )
        x_2_1 = self.block_2_1( cat( self.up_3( x_3_1 ), x_2_0 ) )
        x_1_1 = self.block_1_1( cat( self.up_2( x_2_1 ), x_1_0 ) )
        x_0_1 = self.block_0_1( cat( self.up_1( x_1_1 ), x_0_0 ) )
        
        return self.out_0( x_0_1 ), self.out_1( x_1_1 ), self.out_2( x_2_1 ), \
               self.out_3( x_3_1 ), self.out_4( x_4_1 ), self.out_5( x_5_1 )
















