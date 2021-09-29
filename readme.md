# Rock-Paper-Scissors Recognition

Application of Computer Vision to the classic game Rock-Paper-Scissors.
This repository includes dataset analysis and modeling part of the task. 

The model will be deployed as a JavaScript application on <a href="https://www.romaglushko.com/">my website</a>. 

Dataset: https://www.kaggle.com/frtgnn/rock-paper-scissor

## Installation

```bash
poetry install
cd data
kaggle datasets download --unzip frtgnn/rock-paper-scissor
```
## Modeling

The best scores I was able to achieve with the following configs:

- Fully Freezed MobileNetV2 + RMSProp + L2Regularization(0.01) - ?
- Fully Freezed MobileNetV2 + AdamW(weight_decay: 0.01) - ?


## References

- https://keras.io/guides/transfer_learning/
- https://www.tensorflow.org/tutorials/images/transfer_learning
- https://ruder.io/transfer-learning/
- http://www.laurencemoroney.com/rock-paper-scissors-dataset/