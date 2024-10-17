
from itertools import product
import torch.nn.functional as F
import torch.nn as nn
#import module as MD
import numpy as np
import einops
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
        self.block_4_0 = res2_cm_block_v2( bc*8, mc, [1,3,3], [1,16,16] ) # 16
        self.block_5_0 = res2_cm_block_v2( mc  , mc, [1,3,3], [1,8,8] ) # 8
        self.block_6_0 = res2_cm_block_v2( mc  , mc, [1,3,3], [1,4,4] ) # 4
  
        ## Decoder
        self.block_0_1 = res2_block_4( bc*2    , bc  , [1,3,3], [1,13,13], groups_2=16 )
        self.block_1_1 = res2_block_4( bc*4    , bc*2, [1,3,3], [1,11,11], groups_2= 8 )
        self.block_2_1 = res2_block_4( bc*8    , bc*4, [1,3,3], [1, 9, 9], groups_2= 4 )
        self.block_3_1 = res2_block_4( bc*16   , bc*8, [1,3,3], [1, 7, 7], groups_2= 2 )
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


class res2_block_4( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size_1, kernel_size_2, groups_1=1, groups_2=1, norm='in', ln_shape='None' ):
        super( res2_block_4, self ).__init__()
        kernel_size_1 = np.array( kernel_size_1 ).astype( np.int32 )
        kernel_size_2 = np.array( kernel_size_2 ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        if norm == 'bn':
            nnNorm = nn.BatchNorm3d
        elif norm == 'in':
            nnNorm = nn.InstanceNorm3d
        elif norm == 'ln':
            nnNorm = nn.LayerNorm
        else:
            raise Exception( f'norm={norm}, but there is no such value')
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size_1, 
                                     padding     = (kernel_size_1//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            if norm == 'ln':
                self.norm_0 = nnNorm( ln_shape )
            else:
                self.norm_0 = nnNorm( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size_1, 
                                 padding     = (kernel_size_1//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                                 groups      = groups_1,
                               )
        if norm == 'ln':
            self.norm_1 = nnNorm( ln_shape )
        else:
            self.norm_1 = nnNorm( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size_2, 
                                 padding     = (kernel_size_2//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                                 groups     = groups_2,
                               )
        if norm == 'ln':
            self.norm_2 = nnNorm( ln_shape )
        else:
            self.norm_2 = nnNorm( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        x_1 = self.non_line( self.norm_1( self.conv_1( x   ) ) )
        x_2 = self.norm_2( self.conv_2( x_1 ) )
        return self.non_line( torch.add(  x_2, x ) )
        
class res2_block( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size, norm='in', ln_shape=None ):
        super( res2_block, self).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        if norm == 'bn':
            nnNorm = nn.BatchNorm3d
        elif norm == 'in':
            nnNorm = nn.InstanceNorm3d
        elif norm == 'ln':
            nnNorm = nn.LayerNorm
        else:
            raise Exception( f'norm={norm}, but there is no such value')
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size, 
                                     padding     = (kernel_size//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            if norm == 'ln':
                self.norm_0 = nn.LayerNorm( ln_shape )
            else:
                self.norm_0 = nnNorm( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        if norm == 'ln':
            self.norm_1 = nn.LayerNorm( ln_shape )
        else:
            self.norm_1 = nnNorm( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        if norm == 'ln':
            self.norm_2 = nn.LayerNorm( ln_shape )
        else:
            self.norm_2 = nnNorm( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        x_1 = self.non_line( self.norm_1( self.conv_1( x ) ) )
        x_2 = self.conv_2( x_1 )
        return self.non_line( torch.add( self.norm_2( x_2 ), x ) )
    
class res2_cm_block_v2( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size, shape ):
        super( res2_cm_block_v2, self).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size, 
                                     padding     = (kernel_size//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            #self.norm_0 = nn.BatchNorm3d( out_channel )
            self.norm_0 = nn.InstanceNorm3d( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        #self.norm_1 = nn.BatchNorm3d( out_channel )
        self.norm_1 = nn.InstanceNorm3d( out_channel )
        ## FFN_1
        hidden_size = np.prod( shape )
        self.FFN_1 = nn.Linear( hidden_size, hidden_size, bias=True )
        self.norm_2 = nn.LayerNorm( hidden_size )
        ## FFN_2
        self.FFN_2 = nn.Linear( hidden_size, hidden_size, bias=True )
        self.norm_3 = nn.LayerNorm( hidden_size )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        #
        x_1 = self.non_line( self.norm_1( self.conv_1( x ) ) )
        #
        h, l, w = x_1.shape[2:]
        x_1 = einops.rearrange( x_1, 'b c h l w -> b c (h l w) ' )
        x_1 = self.norm_2( self.FFN_1( x_1 ) )
        x_2 = self.norm_3( self.FFN_2( x_1 ) )
        x_2 = einops.rearrange( x_2, 'b c (h l w) -> b c h l w ',  h=h, l=l, w=w )
        #
        # theta_x = einops.rearrange( x1, 'b c h l w -> b (h l w) c ' )
        # torch.einsum('ijk,ikl->ijl', [ theta_x, phi_x ] )

        return self.non_line( torch.add( x_2, x ) )

class uconv( nn.Module ):
    def __init__( self, in_channel, out_channel, kernel_size ):
        super( uconv, self).__init__()
        
        self.conv_1 = nn.ConvTranspose3d( in_channel,
                                          out_channel, 
                                          kernel_size = kernel_size, 
                                          stride      = kernel_size
                                        )        
                    
    def forward( self, x ):
        return self.conv_1( x )

class out_conv( nn.Module ):
    def __init__( self, in_channel, out_channel,  ):
        super( out_conv, self ).__init__()    
        self.conv_1 = nn.Conv3d( in_channel, 
                                 out_channel, 
                                 kernel_size = 1, 
                                 stride      = 1, 
                                 padding     = 0, 
                                 bias        = False
                               )
    def forward( self, x ):
        return self.conv_1( x )

def max_pool3d( x, kernel ):
    return F.max_pool3d( x, kernel, kernel )

def cat( x, y ):
    return torch.cat( [ x, y ], dim=1 )

if __name__ == '__main__':
    
    # This is a 2D demo. If you need to use a 3D demo, you'll need to modify 
    # the operators in the model to be 3D as well, rather than just converting 
    # the input data to 3D.
    
    model = LD_UNet( in_channel=1, out_channel=2 )
    x = torch.randn( 1, 1, 1, 256, 256 )
    y = model( x )
    for i in y:
        print( i.shape )
    print( model )
    
    print('Done')



