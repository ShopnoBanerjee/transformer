# Transformer Architecture Overview
---
<img src="https://github.com/user-attachments/assets/186fafe5-7af3-4a65-98fb-d396ffcafe31" height="300" />

---


## Token Representation
- Each token in the input sentence is represented by a vector of size 512.
- If the sentence has N tokens, its shape is (N, 512).
- The dimensions are divided evenly across 8 attention heads, each having a shape of (N, 64).

## Attention Mechanism

### Key, Query, and Value Vectors
- Each head creates its own K, Q, and V vectors with their learned weights. 
- Each head computes attention scores using the Q, K, and V, and then uses softmax to normalize the results.

    The attention mechanism is defined as:
    $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### Why is the Square Root of d Used?
- The term $\left(QK^T\right)$ can produce very large values depending on the dimensions, which could lead the softmax function to produce near-binary outputs. We stabilize this by dividing by $\sqrt{d_k}$, ensuring that the attention scores are scaled properly.

### Why Split the Embedding Space?
- The embedding space is split to speed up computation and allow each head to capture different aspects of the input.

### Concatenation of Attention Heads
- After each attention head calculates its values and outputs the shape (N, 64), they are concatenated into a single block. 
- The outputs are stacked horizontally to form a matrix of shape (N, 512).

## Linear Transformation After Concatenation

- The concatenated outputs are passed through a linear transformation to project them back into the original embedding space.
- **Why is this required?**
  Even though the dimensions are the same, the outputs of the heads have learned in a disjointed manner, which is not optimal. A linear projection helps mix them together, making the output more suitable for subsequent layers.

### Purpose of Linear Transformation
- It ensures that the information from each attention head is combined optimally and aligned with the downstream processing layers (like the feed-forward network). Each head might have captured different features or relationships, but without a proper mixing function, those features might not be useful for the model’s next steps.

### Why Not Just Concatenate and Pass?
- If we just concatenate the outputs, each attention head’s output might contain useful features but also extraneous ones. The linear projection effectively filters and combines the features, helping the network focus on the most useful ones for subsequent layers.

## Residual Connection and Layer Normalization

- The attention output is added to the input embedding through a residual connection, ensuring that the model does not "forget" its original input. This helps with gradient flow and avoids the vanishing gradient problem, making it easier to update weights in earlier layers.
- Layer normalization is applied with the formula:
  $\text{Output} = \frac{\text{Output} - \mu}{\sigma}$
  where $\mu$ is the mean and $\sigma$ is the standard deviation.

## Feed Forward Network (FFN)

- The feed-forward network consists of 2 dense layers with a ReLU activation function in between:
  - The input shape is (N, 512), which is transformed into a higher-dimensional space (N, 2048) to learn more complex relationships and patterns.
  - After passing through ReLU, the data is projected back to the original embedding size (N, 512).

### Why is the FFN Required?
- The attention mechanism focuses on the context of each token in relation to others, but the FFN operates on each token independently.
- It introduces non-linearity, helping the model learn complex relationships between tokens. ReLU activation helps by focusing on meaningful interactions, and the projection back to (N, 512) ensures concise and richer information on the token.

### The Role of Attention and FFN
  - **Question**: If attention can understand relationships between tokens, why use the FFN? For example, in the sentence "There is a cat on the mat":
    - The self-attention mechanism computes attention scores, determining how related two tokens are. For instance, "cat" will attend to "mat" if they have high attention scores.
    - After attention, the output encodes the contextual relationship, indicating that "cat" might carry information about "mat" due to the attention scores. However, attention alone might not capture higher-order relationships or abstract patterns in the data.
    - The FFN helps capture these higher-order relationships, especially linguistic patterns that attention might not directly understand. For instance, the FFN might help the model understand how different relationships between multiple tokens like "cat", "is", "on", and "mat" contribute to the overall meaning.

## Another Residual Connection and Layer Normalization

- After the FFN, another residual connection is added to the output of the attention mechanism, followed by layer normalization.

## Repeating the Process

- The process of attention, FFN, and normalization is repeated for \(N\) layers.

---

## Output of Encoder
- After processing through all encoder layers, the final output is a tensor of shape (N, 512), where each token has a contextually rich representation.

## Decoder Operations

- The decoder performs similar operations to the encoder but includes an additional cross-attention step, where it attends to the encoder's output, incorporating information from the source sequence.

### Masked Self-Attention in Decoder
- The first step in the decoder is a masked self-attention block to ensure that the decoder only attends to tokens generated before it. This helps in generating the next token in the sequence.

### Cross-Attention
- After masked self-attention, the decoder performs cross-attention, attending to the **output of the encoder**. The Q values come from the decoder, while the K and V values come from the encoder output.

### FFN in Decoder
- The decoder applies the feed-forward network to capture more abstract relationships, followed by residual connections and layer normalization.

### Generation of Output Sequence
- After passing through all 6 layers, the decoder produces a sequence of logits that represent the probability of each token in the vocabulary.

### Logits and Probability Distribution
The generation of tokens happens as follows:

$$
\text{logits} = \text{decoder} \times W + b
$$

After generating the logits, the softmax function is applied to the logits vector to convert it into a probability distribution over the vocabulary. A decoding strategy is then used to select the next token.

