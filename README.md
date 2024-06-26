
<h1 align='center'> EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting<br>(WWW 2022)<br>
    [<a href="https://dl.acm.org/doi/abs/10.1145/3485447.3512030">paper</a>] </h1>

Deep learning inspired by differential equations is a recent research trend and has marked the state of the art performance for many machine learning tasks. Among them, time-series modeling with neural controlled differential equations (NCDEs) is considered as a breakthrough. In many cases, NCDE-based models not only provide better accuracy than recurrent neural networks (RNNs) but also make it possible to process irregular time-series. In this work, we enhance NCDEs by redesigning their core part, i.e., generating a continuous path from a discrete time-series input. NCDEs typically use interpolation algorithms to convert discrete time-series samples to continuous paths. However, we propose to i) generate another latent continuous path using an encoder-decoder architecture, which corresponds to the interpolation process of NCDEs, i.e., our neural network-based interpolation vs. the existing explicit interpolation, and ii) exploit the generative characteristic of the decoder, i.e., extrapolation beyond the time domain of original data if needed. Therefore, our NCDE design can use both the interpolated and the extrapolated information for downstream machine learning tasks. In our experiments with 5 real-world datasets and 12 baselines, our extrapolation and interpolation-based NCDEs outperform existing baselines by non-trivial margins.
<p align="center">
  <img align="middle" src="./EXIT1.png" alt="EXIT1"/> 
  The overall architecture of EXIT for Extrapolation
</p>
<p align="center">
  <img align="middle" src="./EXIT0.png" alt="EXIT"/> 
  The overall architecture of EXIT for Interpolation
</p>

### create conda environments
```
conda env create --file  neuralcde.yml
```

### activate conda 
```
conda activate neuralcde
```
Go to Experiments folder

### Train sepsis
```
sh sepsis.sh
```
### Citation
```
@inproceedings{jhin2022exit,
  title={Exit: Extrapolation and interpolation-based neural controlled differential equations for time-series classification and forecasting},
  author={Jhin, Sheo Yon and Lee, Jaehoon and Jo, Minju and Kook, Seungji and Jeon, Jinsung and Hyeong, Jihyeon and Kim, Jayoung and Park, Noseong},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={3102--3112},
  year={2022}
}
```
