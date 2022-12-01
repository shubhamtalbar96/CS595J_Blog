# <img src="img/logo.jpg" width="8%" alt="" align=center /> DiffuSeq

<p align="center"></p>
<table align="center">
    <thead>
        <tr>
            <th align="center">TITLE</th>
            <th align="center">AUTHOR</th>
            <th align="center">DATE</th>
            <th align="center">CATEGORY</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Sequence To Sequence Text Generation With Diffusion Models</td>
            <td align="center">Shubham Talbar</td>
            <td align="center">2022-10-28</td>
            <td align="center">NLP</td>
        </tr>
    </tbody>
</table>
<p></p>
  
<p align = "left">
The following blog is an attempt to simplify and understand the use of Diffusion Models in complex conditional language generation tasks. This article is based on the paper <strong>DIFFUSEQ: Sequence To Sequence Text Generation With Diffusion Models</strong>
</p>
  
<!-- more -->

Paper: <https://arxiv.org/pdf/2210.08933.pdf>

GitHub Repo: <https://github.com/Shark-NLP/DiffuSeq>

Official Codebase for [*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*](https://arxiv.org/abs/2210.08933).

&nbsp;

## What problem does the paper solve?

Diffusion Models have recently emerged as a new paradigm for generative models. These models have had success in domains using continuous signals such as vision and audio. But adapting diffusion models to natural language is difficult due to the discrete non-continuous nature of text. This paper tries to tackle this task by proposing **DiffuSeq** which is a diffusion model designed for sequence-to-sequence text generation tasks. The authors evaluate the performance of **DiffuSeq** over a wide variety of SeqToSeq tasks and reported that it infact performed better than those and a state-of-the-art model based on pre-trained language models.

&nbsp;

## What are diffusion models?

Existing generative models such as Generative Adversarial Networks (Goodfellow et al., 2014), Variational Auto Encoders (Kingma & Welling, 2014) and Flow-based models (Dinh et al., 2017) have shown great success in generating high-quality samples, but each has some limitation of its own. GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAE relies on a surrogate loss. Flow models have to use specialized architectures to construct reversible transform.   

Diffusion models are inspired by non-equilibrium thermodynamics. They define a markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as original data). 

&nbsp;

<p align = "center">
<img src="img/diffusionExample.png" width="95%" alt="" align=center />
</p>

<p align = "center">
<strong>Overview of different types of generative models</strong>
</p>

&nbsp;

The key concept in Diffusion Modelling is that if we could build a learning model which can learn the systematic decay of information due to noise, then it should be possible to reverse the process and therefore, recover the information back from the noise. This concept is similar to VAEs in the way that it tries to optimize an objective function by first projecting the data onto the latent space and then recovering it back to the initial state. However, instead of learning the data distribution, the system aims to model a series of noise distributions in a Markov Chain and “decodes” the data by undoing/denoising the data in a hierarchical fashion.

&nbsp;

## Denoising Diffusion Models

The idea of denoising diffusion model has been around for a long time. It has its roots in Diffusion Maps concept which is one of the dimensionality reduction techniques used in Machine Learning literature. It also borrows concepts from the probabilistic methods such as Markov Chains which has been used in many applications. The original Denoising Diffusion method was proposed in Sohl-Dickstein et al. A denoising diffusion modeling is a two step process: the forward diffusion process and the reverse process or the reconstruction. In the forward diffusion process, gaussian noise is introduced successively until the data becomes all noise. The reverse reconstruction process undoes the noise by learning the conditional probability densities using a neural network model. An example depiction of such a process can be visualized below.


&nbsp;

<p align = "center">
<img src="img/reconstruction.webp" width="95%" alt="" align=center />
</p>

<p align = "center">
<strong>Process of Denoising Diffusion Probabilistic Model</strong>
</p>

&nbsp;

## Why and What is DiffuSeq?

DiffuSeq is an extended vanilla diffusion model to learn conditional text generation, concerning the model architecture and the training objective. 

SEQ2SEQ is an essential setting in NLP that covers a wide range of important tasks such as open-ended sentence generation, dialogue, paraphrasing, and text style transfer. This paper proposes **DiffuSeq**, a classifier-free **diffusion model** that supports SEQ2SEQ text generation tasks. By modeling the conditional probability of the target sentence **w** given context **x** using one single model, one advantage of DIFFUSEQ is that this paradigm allows a complete model to fit data distribution and utilize conditional guidance, rather than depending on a separate classifier.

To establish the effectiveness of **DiffuSeq**, the authors conduct experiments on four SEQ2SEQ tasks. Compared to autoregressive (AR) and non-autoregressive (NAR) models, which suffer from the “degeneration” problem (Holtzman et al., 2019) and rely on decoding strategies, DIFFUSEQ can achieve considerable sentence-level diversity without sacrificing the quality.

&nbsp;

<p align = "center">
<img src="img/diffuseq-process.png" width="95%" alt="" align=center />
</p>

<p align = "center">
<strong>The diffusion process of our conditional diffusion language model DiffuSeq</strong>
</p>

&nbsp;

## Experiments

The authors conduct experiments to validate the effectiveness of DIFFUSEQ on four different tasks, against six strong AR/NAR baselines. 

**Tasks and Datasets**

## Significant Contributions

- The proposed __*DiffuSeq*__ as a conditional language model is trained end-to-end in a classifier-free manner.
- The authors have established a theoretical connection among AR, NAR and __*DiffuSeq*__ models (refer to the original paper).
- __*DiffuSeq*__ is a powerful model for text generation, matching or even surpassing competitive AR, iterative NAR, and large-PLMs on quality and diversity.

The study addresses promising achievements by such a new sequence-to-sequence learning paradigm.

<p align = "center">
<img src="img/result-1.png" width="80%" alt="" align=center />
</p>
<p align = "center">
<img src="img/result-2.png" width=80%" alt="" align=center />
</p>

&nbsp;

## Conclusion
Important questions addressed in this blog:

1. What problem does the paper solve specifically?
2. What are diffusion models?
3. What are forward and reverse process in Diffusion Models?
4. Why and how is DiffuSeq effective?

DIFFUSEQ tackles SEQ2SEQ tasks in a diffusion way, which contains the strong potential to achieve better generation quality and diversity trade-off. In other words, DIFFUSEQ is capable of producing highly diverse sentences without sacrificing much of the quality. The capability enables favorable characteristics of DIFFUSEQ to further enhance the quality of final results, by leveraging a minimum Bayes risk decoding algorithm. Besides, the author prove theoretically that DiffuSeq model is a powerful extension of iterativeNAR model. The empirical results demonstrate that DIFFUSEQ is also a powerful model for text generation, matching or even surpassing competitive autoregressive, iterative non-autoregressive, and large-scale pre-trained models on quality and diversity. Given the limited progress of current diffusion models on text generation, our study addresses promising achievements by such a new sequence-to-sequence learning paradigm.
                                             
&nbsp;
                                             
## Reference 
                                             
```
@article{gong2022diffuseq,
  title={DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models},
  author={Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2210.08933},
  year={2022}
}
```

```
@article{diffusionrafideasy,
    title   = "Diffusion Models Made Easy",
    author  = "J. Rafid Siddiqui",
    journal = "[lilianweng.github.io](https://azad-wolf.medium.com/)",
    year    = "2022",
    month   = "May",
    url     = "https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da"
}                                            
```                                          

```
@article{weng2021diffusion,
    title   = "What are diffusion models?",
    author  = "Weng, Lilian",
    journal = "lilianweng.github.io",
    year    = "2021",
    month   = "Jul",
    url     = "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/"
}                                            
```
                                                                               
                                                                               
                                             
                                             
                                             
