# Vocabulary
The following are the models that get applied on each shot. They are agnostic to whether overall we are training on single shot,
entire measurement point (MP) or just the central grid (4x4).
The different approaches correspond simply to different batch sizes for the inputs. My use of the term "batch" here is not to be 
confused with TensorFlow usage - i.e. model.fit(batch_size=x) - if the model gets in input the entire MP then _one_ input (for TensorFlow)
is going to be a batch of 64 shots.

### Shot-level Models
* dnn_0: Fully Connected model with 0 hidden layers. Output is softmax with 8 nodes.
* dnn_1: Fully Connected model with 1 hidden layers. Output is softmax with 8 nodes.

	           Input --> 256 relu layer --> dropout (p=0.01) --> output
* dnn_3: Fully Connected model with 3 hidden layers. Output is softmax with 8 nodes. 

	           LAYER(X) = X relu layer --> dropout (p=0.01)
	
	           Input --> LAYER(512) --> LAYER(256) --> LAYER(128) --> output
* dnn_5: Fully Connected model with 5 hidden layers. Output is softmax with 8 nodes.

	           Input --> LAYER(1024) --> LAYER(1024) --> LAYER(512) --> LAYER(256) --> LAYER(128) --> output



MP: model gets in input a batch of 64 shots. This implies that some sort of pooling is applied on top of the model and the true
	output (from which the gradient flows) is what comes out of the pooling layer.
shot: model gets a single shot as input
central_grid: model gets a batch of 16 inputs corresponding to the 4x4 central grid



### Pooling Layers
* average: Average Pooling on the batch axis
* max: Max Pooling on the batch axis
* dnn: output of the shot model is flattened and an 8 node FCL softmax is applied on top.
* dnn_split: just like dnn, but model has 2 outputs; one before the flattening and one which is the default output. Both outputs
	get labels, losses and gradients.
* major_vote: The model below a gets a "shot" input (see above) and is trained with that. After training the model is evaluated on the test
	set by taking major vote. Notice that training set is a set of shot inputs, whilst test set is a set of MP inputs.
* standard: "shot" input and no pooling. This model has a set of shots as test_set instead of having MPs.
