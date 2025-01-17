from absl import app
import time

import numpy as np
import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_classifier

from shark.shark_trainer import SharkTrainer


tf.random.set_seed(0)
vocab_size = 100
NUM_CLASSES = 5
SEQUENCE_LENGTH = 512
BATCH_SIZE = 1
# Create a set of 2-dimensional inputs
bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
]


class BertModule(tf.Module):

    def __init__(self):
        super(BertModule, self).__init__()
        dict_outputs = False
        test_network = networks.BertEncoder(vocab_size=vocab_size,
                                            num_layers=2,
                                            dict_outputs=dict_outputs)

        # Create a BERT trainer with the created network.
        bert_trainer_model = bert_classifier.BertClassifier(
            test_network, num_classes=NUM_CLASSES)
        bert_trainer_model.summary()

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m = bert_trainer_model
        self.m.predict = lambda x: self.m.call(x, training=False)
        self.predict = tf.function(input_signature=[bert_input])(self.m.predict)
        self.m.learn = lambda x, y: self.m.call(x, training=False)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

    @tf.function(input_signature=[
        bert_input,  # inputs
        tf.TensorSpec(shape=[BATCH_SIZE], dtype=tf.int32)  # labels
    ])
    def forward(self, inputs, labels):
        with tf.GradientTape() as tape:
            # Capture the gradients from forward prop...
            probs = self.m(inputs, training=True)
            loss = self.loss(labels, probs)

        # ...and use them to update the model's weights.
        variables = self.m.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


if __name__ == "__main__":
    predict_sample_input = [
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH)),
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH)),
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH))
    ]
    sample_input_tensors = [tf.convert_to_tensor(val, dtype=tf.int32) for val in predict_sample_input]
    num_iter = 10
    shark_module = SharkTrainer(
        BertModule(),
        (sample_input_tensors,
         tf.convert_to_tensor(np.random.randint(5, size=(BATCH_SIZE)), dtype=tf.int32)))
    shark_module.set_frontend("tensorflow")
    shark_module.compile()
    start = time.time()
    print(shark_module.train(num_iter))
    end = time.time()
    total_time = end - start
    print("time: " + str(total_time))
    print("time/iter: " + str(total_time / num_iter))
