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

## Example 01
Origin article (MNIST by TensorFlowKit you can finde here)[https://www.octadero.com/2017/11/16/mnist-by-tensorflowkit/].

I took “Hello World!”  in the universe of neural networks as an example, a task for systematization of MNIST images. MNIST dataset includes thousands of images of handwritten numbers, the size of each image is 28×28 pixels. So, we have ten classes that are neatly divided into 60 000 images for educating and 10 000 images for testing. Our task is to create a neural network that is able to classify an image and determine the class it belongs to (out of 10 classes).


Before you can start working with TensorFlowKit, you need to install TensorFlow On Mac OS, you can use the brew package manager:
```
$ brew install libtensorflow
```

Assembly for [Linux is available her](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.4.0.tar.gz).

Let’s create a Swift project and add a dependency:
```
dependencies: [
.package(url: "https://github.com/Octadero/TensorFlow.git", from: "0.0.7")
]
```
Now we should prepare the MNIST dataset.

I have written a Swift package for working with the MNIST dataset that you can find here. This package will download the dataset to a temporary folder, unpack it, and represent it as ready-to-use classes.

For example:
```
dataset = MNISTDataset(callback: { (error: Error?) in
print("Ready")
})
```
Now let’s create the required operation graph.

The space and subspace of the calculation graph is called scope and can have its own name. We’ll provide two vectors for the network input. The first one contains the images represented as a 784 high-dimension vector (28×28 px). So, each component of the x vector will contain a Float from 0.0-1.0 value that corresponds to the color of the pixel on the image. The second vector will be an encrypted matching class (see below), where the corresponding component 1 matches the class number.  In the following example it’s class 2.

```
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]
```
As input parameters will change during the educative process, let’s create a placeholder to refer to them.

```
/// Input sub scope
let inputScope = scope.subScope(namespace: "input")
let x = try inputScope.placeholder(operationName: "x-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 784]))
let yLabels = try inputScope.placeholder(operationName: "y-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 10]))
```

That’s how Input looks on the graph:
![image-1](http://storage.googleapis.com/api.octadero.com/articles/01/graph_2.png)

That is our input layer. Now let’s create weights (connections) between the input and hidden layer.

```
let weights = try weightVariable(at: scope, name: "weights", shape: Shape.dimensions(value: [784, 10]))
let bias = try biasVariable(at: scope, name: "biases", shape: Shape.dimensions(value: [10]))
```

We will create a variable operation in the graph, because the weights and bases will be customized during the educative process. Let’s initialize them using a tensor filled with nulls.

![image-2](http://storage.googleapis.com/api.octadero.com/articles/01/graph_3.png)
Now let’s create a hidden layer that will perform such primitive operation as (x * W) + b. [This operation multiplies vector x (dimension 1×784) by matrix W (dimension 784×10) and adds basis](https://www.octadero.com/2017/02/09/matrix-multiplication-ways/).

In our case the hidden layer is the output layer (the task of the “Hello World!” level), that’s why we need to analyze the output signal and decide the winner.  To do that, we should use the softmax operation.

![image-3](http://storage.googleapis.com/api.octadero.com/articles/01/graph_5.png)

I suggest to take our neural network as a complicated function in order to better understand what I will be talking about hereafter. We input vector x (representing the image) to our function. In the output we get a vector that shows the probability of the input vector belonging to each of the available classes.

Now let’s take a natural logarithm of the received probability for each class and multiply it by the value of the vector of the right class neatly passed in the very beginning (yLabel). This way we will get the error value and use it to “judge” the neural network. The figure below demonstrates two samples. In the first sample, for class 2 the error value is 2.3, and in the second sample, for class 1 the error value is 0.
![image-4](http://storage.googleapis.com/api.octadero.com/articles/01/TensorFlow%26Swift-3.png)

```
let log = try scope.log(operationName: "Log", x: softmax)
let mul = try scope.mul(operationName: "Mul", x: yLabels, y: log)
let reductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(1)]), as: "reduction_indices").defaultOutput
let sum = try scope.sum(operationName: "Sum", input: mul, reductionIndices: reductionIndices, keepDims: false, tidx: Int32.self)
let neg = try scope.neg(operationName: "Neg", x: sum)
let meanReductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(0)]), as: "mean_reduction_indices").defaultOutput
let cross_entropy = try scope.mean(operationName: "Mean", input: neg, reductionIndices: meanReductionIndices, keepDims: false, tidx: Int32.self)
```

What to do next?
If talking mathematical language, we have to minimize the target function. To do that, the gradient descent method can be used. If it may become necessary, I will try to describe this method in another article.

So, we should calculate how to correct each of the weighs (components of the W matrix) and the basis vector b, so that the neural network would make smaller error when receiving similar input data. In the context of math, we should find the partial derivatives of the output node by the values of all intermediate nodes. The symbolic gradients we’ve got allow us to “move” the values of the W and b variables according to the extent it affected the result of the previous calculations.

TensorFlow Magic

The thing is that TensorFlow can perform all (however, not all at the very moment) these complicated calculations automatically by analyzing the graph we created.

```
let gradientsOutputs = try scope.addGradients(yOutputs: [cross_entropy], xOutputs: [weights.variable, bias.variable])
```
After this operation call, TensorFlow will create about fifty more operations.

![image-5](http://storage.googleapis.com/api.octadero.com/articles/01/graph_6.png)

```
let _ = try scope.applyGradientDescent(operationName: "applyGradientDescent_W",
`var`: weights.variable,
alpha: learningRate,
delta: gradientsOutputs[0],
useLocking: false)
```

That’s it – the graph is ready!

![image-5](http://i0.wp.com/storage.googleapis.com/api.octadero.com/articles/01/graph_6%402x.png)

As I said, TensorFlow separates the model and calculations. That’s why the graph we created is only a model for performing calculations. We can use Session to start the calculation process. Let’s prepare data from the dataset, place it to tensors, and run the session.

```
guard let dataset = dataset else { throw MNISTTestsError.datasetNotReady }
guard let images = dataset.files(for: .image(stride: .train)).first as? MNISTImagesFile else { throw MNISTTestsError.datasetNotReady }
guard let labels = dataset.files(for: .label(stride: .train)).first as? MNISTLabelsFile else { throw MNISTTestsError.datasetNotReady }
let xTensorInput = try Tensor(dimensions: [bach, 784], values: xs)
let yTensorInput = try Tensor(dimensions: [bach, 10], values: ys)
```
It is necessary to run the session several times to let it recalculate the value several times.

```
for index in 0..<1000 {

let resultOutput = try session.run(inputs: [x, y],
values: [xTensorInput, yTensorInput],
outputs: [loss, applyGradW, applyGradB],
targetOperations: [])

if index % 100 == 0 {
let lossTensor = resultOutput[0]
let gradWTensor = resultOutput[1]
let gradBTensor = resultOutput[2]
let wValues: [Float] = try gradWTensor.pullCollection()
let bValues: [Float] = try gradBTensor.pullCollection()
let lossValues: [Float] = try lossTensor.pullCollection()
guard let lossValue = lossValues.first else { continue }
print("\(index) loss: ", lossValue)
lossValueResult = lossValue
print("w max: \(wValues.max()!) min: \(wValues.min()!) b max: \(bValues.max()!) min: \(bValues.min()!)")

}
}
```

The error range is shown after every 100 operations. In the next article, I will tell you how to calculate the accuracy of our network and how to visualize it using the means of TensorFlowKit.

