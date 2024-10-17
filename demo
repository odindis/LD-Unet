'''
This is a 2D demo. If you need to use a 3D demo, you'll need to modify 
the operators in the model to be 3D as well, rather than just converting 
the input data to 3D.
'''
  
import model 
import torch
  
model = model.LD_UNet( in_channel=1, out_channel=2 )
x = torch.randn( 1, 1, 1, 256, 256 )
y = model( x )
for i in y:
    print( i.shape )
print( model )

print('Done')


  
