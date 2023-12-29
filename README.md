**Model Selection**

In the Speech to Text conversion using the Transformer based architecture due to the hight performance in pretrained model.The Transformer's self-attention mechanism provides a robust framework for processing both audio and text data simultaneously.

**Data Preprocessing**

Using the model is trained by open-ai whisper-large model.

**Training Methodology**

We used PyTorch for implementing the Transformer model. The training pipeline included data loading using DataLoader, model initialization with custom configurations, and training with Adam optimizer and a learning rate of 1e-4.

**Evaluation**

Word Error Rate (WER)
we can check the accracy by using Word Error Rate by the splitting the words and compare.
