# 01_MNIST

## Build and run

### TensorFlow
Do not forget install libtensorflow.
On mac OS:
```
brew install libtensorflow
```
Linux:
```
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.4.0.tar.gz
```
Also, you can install it from sources, [how to install TensorFlow from sources you can find here](https://www.octadero.com/2017/08/27/tensorflow-c-environment/).


### Xcode
To generate xcode project file you can call:

```
swift package -Xlinker -rpath -Xlinker /usr/local/Cellar/libtensorflow/1.4.0/lib generate-xcodeproj
```
You have to set rpath to you tensorflow library folder.

Build

```
swift build -Xlinker -rpath -Xlinker /usr/local/Cellar/libtensorflow/1.4.0/lib
```

## Example
More details about [TensorFlow + Swift + MNIST here.](https://www.octadero.com/2017/11/11/hello-mnist/)
