#%%
from model import Deeplabv3

deeplab_model = Deeplabv3(input_shape=(512, 512, 3), classes=4)

deeplab_model.summary()

# %%
