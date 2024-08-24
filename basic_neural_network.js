const tf=require('@tensorflow/tfjs')

//now lets create a basic neural network 
//first one using the sequential model 
/*
const model=tf.sequential();
model.add(tf.layers.dense({units:32,inputShape:[5]}));
model.add(tf.layers.dense({units:10,activation:"relu"}));

console.log(model.outputs[0].shape);
*/

//now we try to make another model 
/*
const input=tf.input({shape:[5]});

const dense_layer1=tf.layers.dense({units:10,activations:'relu'});

const dense_layer2=tf.layers.dense({units:5,activations:'relu'});

const output=dense_layer2.apply(dense_layer1.apply(input)) 
//apply dense layer 2 to dense layer 1 output 

//lets wrap the model 
const model=tf.model({inputs:input,outputs:output});

model.predict(tf.ones([4,5])).print();



// now lets create  a neural network in another way 

const x=tf.input({shape:[32]});

const y=tf.layers.dense({units:3,activation:'softmax'}).apply(x);

const model=tf.model({inputs:x,outputs:y});

const inputTensor=tf.ones([3,32])
model.predict(tf.ones([3,32])).print()
*/

