'''
This is a 2D demo. If you need to use a 3D demo, you'll need to modify 
the operators in the model to be 3D as well, rather than just converting 
the input data to 3D.
'''
  
import model 
import torch

in_channel = 1
out_channel = 2
batch_size = 3
image_shape = [1,256,256]

model = model.LD_UNet( in_channel=in_channel, out_channel=out_channel )
x = torch.randn( batch_size, in_channel, image_shape[0], image_shape[1], image_shape[2] )
y = model( x )
for i in y:
    print( i.shape )
print( model )

print('Done')


  
