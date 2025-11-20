# pico-llm
ML pic-llm assignment

Changes made as of 18th Nov
1. Added the K-Gram M LP Sequence Class, and added some comments on how it is very slow, and within expectations
2. Added the Nucleus Sampling manually from the branch, we can merge it later
3. Added the TransformerBlock class, and some other helper class such as Attention Heads etc
4. Added some helper comments, and some "starter CLI options" using which u can run the code in Terminal

Changes made as of 20th Nov
1. Made changes to the "train_one_model()" function to include some stuff like AdamW, warmup and decay to the learning rate etc (to prevent exploding gradients).
2. Above improved the final epoch loss of the transformer by a significant margin
3. Changed some basic starter numbers when making the Transformer model
4. Wrote a separate function which can overfit any model it receives
5. Wrote a small section which makes plots for the normal and overfit methods of the model
