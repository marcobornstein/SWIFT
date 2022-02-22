

# Codebase Ideas & To Do
1) The test accuracy comparison can be improved potentially, since it may not be truly helping so much. The reason for this belief, is that while the worst worker will do more sgd updates, in the same time it does those, other workers may do equally many sgd updates. The only difference is that the other workers communicate with each other sooner. This does make sense, but need to think about more as well as other alternatives...
2) Play around with the max sgd hyperparameter
3) MOVE THE LEARNING RATE ADJUSTMENT OUTSIDE THE TRAINING LOOP
4) Maybe send the consensus model only after the 5th epoch?