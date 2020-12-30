from transformers import Transformer
import tensorflow as tf


# --------- model config ---------
EPOCHS = 20

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = 1e4 + 2
target_vocab_size = 1e4 + 2
dropout_rate = 0.1

learning_rate = 1e-4    # recommend to use dynamic learning rate


# --------- save model config ---------
checkpoint_dir = './ckpt'
checkpoint_max_to_keep = 10

output_dir = './output'


# --------- loss function ---------
# TODO


# --------- transformer ---------
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# --------- optimizer ---------
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


# --------- checkpoint & manager ---------
ckpt = tf.train.Checkpoint(
    optimizer=optimizer,
    transformer=transformer
)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_dir, max_to_keep=checkpoint_max_to_keep)


# --------- define training step ---------
def train_step(inp, tar):
    pass

# --------- define training pipeline ---------


# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------
# ---------  ---------