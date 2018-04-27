//
//  main.swift
//  TensotFlowTest
//
//  Created by Volodymyr Pavliukevych on 4/27/18.
//  Copyright © 2018 Octadero. All rights reserved.
//
import Foundation
import TensorFlow

struct MLPClassifier {
    var w1 = Tensor<Float>(shape: [2, 4], scalars: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    var w2 = Tensor<Float>(shape: [4, 1], scalars: [0.4, -0.5, -0.5, 0.4])
    var b1 = Tensor<Float>([0.2, -0.3, -0.3, 0.2])
    var b2 = Tensor<Float>(shape: [1, 1], scalars: [0.4])

    func prediction(for x: Tensor<Float>) -> Tensor<Float> {
        // The ⊗ operator performs matrix multiplication.
        let o1 = tanh(x ⊗ w1 + b1)
        return tanh(o1 ⊗ w2 + b2)
    }
}

let input = Tensor<Float>(shape: [1, 2], scalars: [0.2, 0.8])
let classifier = MLPClassifier()
let prediction = classifier.prediction(for: input)
print("Result: ", prediction)

