/* Copyright 2017 The Octadero Authors. All Rights Reserved.
 Created by Volodymyr Pavliukevych on 2017.
 
 Licensed under the Apache License 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 https://github.com/Octadero/Examples/02_MNIST/blob/master/LICENSE
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
/*
 IT IS NOT CORRECT WAY TO CALC ACCURACY ON TRAIN DATA
 BUT THE MAIN GOAL OF THAT EXAMPLE IS VISUALIZATION
 SO RIGHT WAY OF ACCURACY IS GOAL FOR OTHER EXAMPLE
 DO NOT DO IN THAT WAY
*/

import Foundation
import CAPI
import Proto
import TensorFlowKit
import MNISTKit

public enum NetworkError: Error {
    case operationNotFound(name: String)
    case datasetNotReady
    case other(message: String)
}

public class Network {
    var dataset: MNISTDataset?
    var fileWriter: FileWriter?

    public init() {
        loadDataset { (error) in
            if let error = error {
                print(error.localizedDescription)
                exit(0)
            } else {
                //MARK: Create a multilayer model.
                do {
                    let scope = try self.buildGraph()
                    try self.learn(scope: scope)
                } catch {
                    print(error.localizedDescription)
                }
                exit(0)
            }
        }
    }
    
    //MARK: - weight and biases
    /// Create a bias variable with appropriate initialization.
    func biasVariable(at scope: Scope, name: String, shape: Shape) throws -> (output: Output, variable: Output) {
        let scope = scope.subScope(namespace: name)
        
        let biasConst = try scope.addConst(tensor: Tensor(shape: shape, values: Array<Float>(repeating: 0.000, count: Int(shape.elements ?? 0))), as: "zero").defaultOutput
        let bias = try scope.variableV2(operationName: "Variable", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try scope.assign(operationName: "Assign", ref: bias, value: biasConst, validateShape: true, useLocking: true)
        let read = try scope.identity(operationName: "read", input: bias)
        return (read, bias)
    }
    
    /// We can't initialize these variables to 0 - the network will get stuck.
    /// Create a weight variable with appropriate initialization.
    func weightVariable(at scope: Scope, name: String, shape: Shape) throws -> (output: Output, variable: Output) {
        let scope = scope.subScope(namespace: name)
        
        let zeros = try scope.addConst(tensor: Tensor(shape: shape, values: Array<Float>(repeating: 0.000, count: Int(shape.elements ?? 0))), as: "zero").defaultOutput
        let weight = try scope.variableV2(operationName: "Variable", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try scope.assign(operationName: "Assign", ref: weight, value: zeros, validateShape: true, useLocking: true)
        let read = try scope.identity(operationName: "read", input: weight)
        return (read, weight)
    }
    
    //MARK: - Neuron (W * x) + b
    func neuron(at scope: Scope, name: String, x: Output, w: Output, bias: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        let matMult = try scope.matMul(operationName: "MatMul", a: x, b: w, transposeA: false, transposeB: false)
        let preactivate = try scope.add(operationName: "add", x: matMult, y: bias)
        return preactivate
    }
    
    /// Main build graph function.
    func buildGraph() throws -> Scope {
        let scope = Scope()
        let summary = Summary(scope: scope)

        //MARK: Input sub scope
        let inputScope = scope.subScope(namespace: "input")
        let x = try inputScope.placeholder(operationName: "x-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 784]))
        let yLabels = try inputScope.placeholder(operationName: "y-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 10]))
        
        let weights = try weightVariable(at: scope, name: "weights", shape: Shape.dimensions(value: [784, 10]))
        let bias = try biasVariable(at: scope, name: "biases", shape: Shape.dimensions(value: [10]))
        
        let neuron = try self.neuron(at: scope, name: "layer", x: x, w: weights.output, bias: bias.output)
        let softmax = try scope.softmax(operationName: "Softmax", logits: neuron)
        
        let log = try scope.log(operationName: "Log", x: softmax)
        let mul = try scope.mul(operationName: "Mul", x: yLabels, y: log)
        let reductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(1)]), as: "reduction_indices").defaultOutput
        let sum = try scope.sum(operationName: "Sum", input: mul, reductionIndices: reductionIndices, keepDims: false, tidx: Int32.self)
        let neg = try scope.neg(operationName: "Neg", x: sum)
        
        let meanReductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(0)]), as: "mean_reduction_indices").defaultOutput
        let cross_entropy = try scope.mean(operationName: "Mean", input: neg, reductionIndices: meanReductionIndices, keepDims: false, tidx: Int32.self)
        
        let gradientsOutputs = try scope.addGradients(yOutputs: [cross_entropy], xOutputs: [weights.variable, bias.variable])
        
        let learningRate = try scope.addConst(tensor: try Tensor(scalar: Float(0.05)), as: "learningRate").defaultOutput
        
        let _ = try scope.applyGradientDescent(operationName: "applyGradientDescent_W",
                                               `var`: weights.variable,
                                               alpha: learningRate,
                                               delta: gradientsOutputs[0],
                                               useLocking: false)
        
        let _ = try scope.applyGradientDescent(operationName: "applyGradientDescent_B",
                                               `var`: bias.variable,
                                               alpha: learningRate,
                                               delta: gradientsOutputs[1],
                                               useLocking: false)
        
        // Accuracy
        let firstDimension = try scope.addConst(tensor: Tensor(scalar: Int(1)), as: "dimension_1").defaultOutput
        let firstArgMax = try scope.argMax(operationName: "ArgMax_1", input: softmax, dimension: firstDimension, tidx: Int32.self, outputType: Int64.self)
        
        let secondDimension = try scope.addConst(tensor: Tensor(scalar: Int(1)), as: "dimension_2").defaultOutput
        let secondArgMax = try scope.argMax(operationName: "ArgMax_2", input: yLabels, dimension: secondDimension, tidx: Int32.self, outputType: Int64.self)
        let correctPrediction = try scope.equal(operationName: "Equal", x: firstArgMax, y: secondArgMax)
        let cast = try scope.cast(operationName: "Cast", x: correctPrediction, srcT: Bool.self, dstT: Float.self)
        let reductionIndicesAccuracy = try scope.addConst(tensor: Tensor.init(dimensions: [1], values: [Int(0)]), as: "reductionIndicesAccuracyConst").defaultOutput
        let accuracy = try scope.mean(operationName: "Accuracy", input: cast, reductionIndices: reductionIndicesAccuracy, keepDims: false, tidx: Int32.self)
        
        // Visualization
        try summary.histogram(output: gradientsOutputs[0], key: "GradientDescentW")
        try summary.histogram(output: gradientsOutputs[1], key: "GradientDescentB")

        try summary.histogram(output: bias.output, key: "bias")
        try summary.histogram(output: weights.output, key: "weights")
        try summary.scalar(output: accuracy, key: "scalar-accuracy")
        try summary.scalar(output: cross_entropy, key: "scalar-loss")
        
        
        let flattenConst = try scope.addConst(values: [Int64(7840)], dimensions: [1], as: "flattenShapeConst")
        
        let imagesFlattenTensor = try scope.reshape(operationName: "FlattenReshape",
                                                    tensor: weights.variable,
                                                    shape: flattenConst.defaultOutput,
                                                    tshape: Int64.self)
        
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 0)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 1)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 2)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 3)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 4)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 5)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 6)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 7)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 8)
        try extractImage(from: imagesFlattenTensor, scope: scope, summary: summary, atIndex: 9)
        
        
        let _ = try summary.merged(identifier: "simple")
        
        // create writer
        try createFileWriter(graph: scope.graph)

        return scope
    }
    
    /// Convert 784 weights per 10 classes to seperet image
    /// doing stride slicing.
    func extractImage(`from` imagesFlattenTensor: Output, scope: Scope, summary: Summary, atIndex index: Int) throws {
        let extractedImage0 = try scope.stridedSlice(operationName: "StridedImage-\(String(index))",
                                                     input: imagesFlattenTensor,
                                                     begin: try scope.addConst(values: [Int(index)], dimensions: [1], as: "Begin-\(String(index))").defaultOutput,
                                                     end: try scope.addConst(values: [Int(28 * 28 * 10)], dimensions: [1], as: "End-\(String(index))").defaultOutput,
                                                     strides: try scope.addConst(values: [Int(10)], dimensions: [1], as: "Strides-\(String(index))").defaultOutput,
                                                     index: Int.self,
                                                     beginMask: 0,
                                                     endMask: 0,
                                                     ellipsisMask: 0,
                                                     newAxisMask: 0,
                                                     shrinkAxisMask: 0)
        
        let shapeImageTensorConst = try scope.addConst(values: Array<Int64>([1, 28, 28, 1]), dimensions: [4], as: "shapeImageTensorConst-\(String(index))")
        let imagesTensor = try scope.reshape(operationName: "Reshape-\(String(index))",
                                             tensor: extractedImage0,
                                             shape: shapeImageTensorConst.defaultOutput,
                                             tshape: Int64.self)
        
        try summary.images(name: "Image-\(String(index))", output: imagesTensor, maxImages: 255, badColor: Summary.BadColor.default)
    }
    
    func learn(scope: Scope) throws {
        let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
        
        guard let wAssign = try scope.graph.operation(by: "weights/Assign") else { throw NetworkError.operationNotFound(name: "weights/Assign") }
        guard let bAssign = try scope.graph.operation(by: "biases/Assign") else { throw NetworkError.operationNotFound(name: "biases/Assign") }
        let _ : [Tensor] = try session.run(inputs: [], values: [], outputs: [], targetOperations: [wAssign, bAssign])
        
        guard let x = try scope.graph.operation(by: "input/x-input")?.defaultOutput else { throw NetworkError.operationNotFound(name: "input/x-input") }
        guard let y = try scope.graph.operation(by: "input/y-input")?.defaultOutput else { throw NetworkError.operationNotFound(name: "input/y-input") }
        
        guard let loss = try scope.graph.operation(by: "Mean")?.defaultOutput else { throw NetworkError.operationNotFound(name: "Mean") }
        guard let applyGradW = try scope.graph.operation(by: "applyGradientDescent_W")?.defaultOutput else { throw NetworkError.operationNotFound(name:  "applyGradientDescent_W") }
        
        guard let applyGradB = try scope.graph.operation(by: "applyGradientDescent_B")?.defaultOutput else { throw NetworkError.operationNotFound(name:  "applyGradientDescent_B") }
        
        guard let dataset = dataset else { throw NetworkError.datasetNotReady }
        guard let images = dataset.files(for: .image(stride: .train)).first as? MNISTImagesFile else { throw NetworkError.datasetNotReady }
        guard let labels = dataset.files(for: .label(stride: .train)).first as? MNISTLabelsFile else { throw NetworkError.datasetNotReady }
        
        
        guard let accuracy = try scope.graph.operation(by: "Accuracy")?.defaultOutput else { throw NetworkError.operationNotFound(name:  "Accuracy") }
        guard let mergedSummary = try scope.graph.operation(by: "MergeSummary-simple")?.defaultOutput else { throw NetworkError.operationNotFound(name:  "MergeSummary-simple") }

        print("Load dataset ...")
        let batch = 2000
        let steps = 5000
        let xs = images.images[0..<batch].flatMap { $0 }
        
        var ys = [Float]()
        labels.labels[0..<batch].forEach { index in
            var label = Array<Float>(repeating: 0, count: 10)
            label[Int(index)] = 1.0
            ys.append(contentsOf: label)
        }
        
        let xTensorInput = try Tensor(dimensions: [batch, 784], values: xs)
        let yTensorInput = try Tensor(dimensions: [batch, 10], values: ys)
        var lossValueResult: Float = Float(Int.max)
        for index in 0..<steps {
            // IT IS NOT CORRECT WAY TO CALC ACCURACY ON TRAIN DATA
            // BUT THE MAIN GOAL OF THAT EXAMPLE IS VISUALIZATION
            // SO RIGHT WAY OF ACCURACY IS GOAL FOR OTHER EXAMPLE
            // DO NOT DO IN THAT WAY
            let resultOutput = try session.run(inputs: [x, y],
                                               values: [xTensorInput, yTensorInput],
                                               outputs: [loss, applyGradW, applyGradB, mergedSummary, accuracy],
                                               targetOperations: [])
            
            if index % 100 == 0 {
                let lossTensor = resultOutput[0]
                let accuracyTensor = resultOutput[4]
                
                let lossValues: [Float] = try lossTensor.pullCollection()
                guard let lossValue = lossValues.first else { continue }

                let accuracyValues: [Float] = try accuracyTensor.pullCollection()
                guard let accuracyValue = accuracyValues.first else { continue }

                
                print("\(index) loss: \(lossValue) accuracy: \(accuracyValue)")
                lossValueResult = lossValue

                let summary = resultOutput[3]
                try fileWriter?.addSummary(tensor: summary, step: Int64(index))
            }
        }
        
        if lossValueResult > 0.2 {
            throw NetworkError.other(message: "Accuracy value not reached.")
        }
    }

    func createFileWriter(graph: Graph) throws {
        guard let writerURL = URL(string: "/tmp/example/") else {
            throw NetworkError.other(message: "Can't compute folder url.")
        }
        fileWriter = try FileWriter(folder: writerURL, identifier: "iMac", graph: graph)
    }
    
    func loadDataset(callback: @escaping (_ error: Error?) -> Void) {
        if dataset == nil {
            dataset = MNISTDataset(callback: callback)
        }
    }
}
