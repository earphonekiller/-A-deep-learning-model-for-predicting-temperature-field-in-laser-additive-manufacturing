# -A-deep-learning-model-for-predicting-temperature-field-in-laser-additive-manufacturing
这个网络是U-net架构的，主要用于图像识别，训练时输入温度场图像和激光路径图像，该神经网络会在训练中逐步建立两张图片的映射关系，最终可以通过激光路径完成对温度场的预测。他可以很方便的改变卷积核大小以及卷积层数，便于讨论在不同的卷积核及卷积层数下模型精度的变化。
main_code文件包含了神经网络的主体，可以将温度场和激光路径图片输入来训练他
batch_size和epoch是分别用来探究batch_size和epoch的大小对模型预测精度的影响的程序
predicted_using是用来调用训练好的神经网络模型的

This network is based on the U-net architecture and is mainly used for image recognition. During training, temperature field images and laser path images are input. The neural network gradually establishes the mapping relationship between the two images during training, and ultimately, the temperature field can be predicted based on the laser path. It can conveniently change the size of the convolution kernel and the number of convolution layers, facilitating the discussion of the changes in model accuracy under different convolution kernels and convolution layers. The main_code file contains the main body of the neural network and can input temperature field and laser path images for training. The batch_size and epoch programs are respectively used to explore the impact of batch_size and epoch size on the prediction accuracy of the model. The predicted_using is used to call the trained neural network model.
