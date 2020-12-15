import keras

model_vgg = keras.applications.VGG16(input_shape=(256, 256, 3),
                                           include_top=False,
                                           weights='imagenet')
model_vgg.trainable = False
model_vgg.summary()

intermediate_model = keras.Model(inputs=model_vgg.input, outputs=model_vgg.get_layer('block2_pool').output)
intermediate_model.summary()

input_tensor = keras.Input(shape=(5, 256, 256, 3))
timeDistributed_layer = keras.layers.TimeDistributed( intermediate_model )(input_tensor)

my_time_model = keras.Model( inputs = input_tensor, outputs = timeDistributed_layer )
my_time_model.summary()
