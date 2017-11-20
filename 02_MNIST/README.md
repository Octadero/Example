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

## Example 02 *Visuzlization*
// IT IS NOT CORRECT WAY TO CALC ACCURACY ON TRAIN DATA
// BUT THE MAIN GOAL OF THAT EXAMPLE IS VISUALIZATION
// SO RIGHT WAY OF ACCURACY IS GOAL FOR OTHER EXAMPLE
// DO NOT DO IN THAT WAY

