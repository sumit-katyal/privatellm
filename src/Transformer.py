import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention

# Define a single transformer block
def transformer_block(input, num_heads, key_dim):
    # Step 1: Multi-head attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input, input)
    # Step 2: Add & Normalize
    attention = LayerNormalization()(attention + input)
    # Step 3: Feedforward network
    dense = Dense(128, activation='relu')(attention)
    # Step 4: Add & Normalize
    output = LayerNormalization()(dense + attention)
    return output

# Input layer
input_layer = Input(shape=(None, 128))  # Sequence length is variable, feature size is 128
# Transformer block
transformer_output = transformer_block(input_layer, num_heads=8, key_dim=64)
# Model definition
model = tf.keras.Model(inputs=input_layer, outputs=transformer_output)

# Model summary
model.summary()