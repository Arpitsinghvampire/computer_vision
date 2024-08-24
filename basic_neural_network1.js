const tf=require('@tensorflow/tfjs')

const input1=tf.input({shape:[10]});
const input2=tf.input({shape:[20]});

const dense1=tf.layers.dense({units:4,activation:'relu'}).apply(input1)

const dense2=tf.layers.dense({units:4,activation:'relu'}).apply(input2)

const dense3=tf.layers.dense({units:4,activation:'relu'}).apply(dense1)
// we now concatenate the layers 

const concatenate=tf.layers.concatenate().apply([dense2,dense3]);

const output=tf.layers.dense({units:2,activations:'softmax'}).apply(concatenate);

// now lets wrap the model
const model=tf.model({inputs:[input1,input2],outputs:[output]})

//lets print the summary of the model

model.summary()

//now lets compile the model

model.compile({optimizer:'sgd',loss:'meanSquaredError'});


model.predict([tf.ones([8,10]),tf.ones([8,20])]).print();


