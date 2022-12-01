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

## Why DiffuSeq?

SEQ2SEQ is an essential setting in NLP that covers a wide range of important tasks such as open-ended sentence generation, dialogue, paraphrasing, and text style transfer. This paper proposes **DiffuSeq**, a classifier-free diffusion model that supports SEQ2SEQ text generation tasks. By modeling the conditional probability of the target sentence **w** given context **x** using one single model, one advantage of DIFFUSEQ is that this paradigm allows a complete model to fit data distribution and utilize conditional guidance, rather than depending on a separate classifier.

To establish the effectiveness of **DiffuSeq**, the authors conduct experiments on four SEQ2SEQ tasks. Compared to autoregressive (AR) and non-autoregressive (NAR) models, which suffer from the “degeneration” problem (Holtzman et al., 2019) and rely on decoding strategies, DIFFUSEQ can achieve considerable sentence-level diversity without sacrificing the quality.

&nbsp;

<p align = "center">
<img src="img/diffuseq-process.png" width="95%" alt="" align=center />
</p>

<p align = "center">
<strong>The diffusion process of our conditional diffusion language model DiffuSeq</strong>
</p>

&nbsp;

## Significant Contributions

- The proposed __*DiffuSeq*__ as a conditional language model is trained end-to-end in a classifier-free manner.
- The authors have established a theoretical
connection among AR, NAR and __*DiffuSeq*__ models (refer to the original paper).
- __*DiffuSeq*__ is a powerful model for text
generation, matching or even surpassing competitive AR, iterative NAR,
and large-PLMs on quality and diversity.

The study addresses promising achievements by such a new
sequence-to-sequence learning paradigm.

<p align = "center">
<img src="img/result-1.png" width="80%" alt="" align=center />
</p>
<p align = "center">
<img src="img/result-2.png" width=80%" alt="" align=center />
</p>

&nbsp;

## Method

The proposed TC approach learns a latent space with smooth temporal dynamics for modeling and creating coherent text. The researchers devised a unique contrastive goal for learning a latent space with Brownian bridge dynamics and then utilized this latent space to create text that keeps local coherence while displaying better global coherence.

The TC text generation pipeline uses the Brownian bridge process to plan a latent trajectory with a start and finish, then conditionally creates sentences that follow this latent plan.

The intuition is simple: The bridge imposes that a positive triplet (eg. three in-order sentences on Boston) makes up a smooth trajectory. A negative triplet should not construct a smooth trajectory (switching middle sentences with one on New York).

After training the encoder, GPT2 is finetuned to decode from past context and the encoded latent plan. At inference, a latent plan is generated by sampling from the bridge and conditionally generating each sentence using the latent plan.


&nbsp;

## Discussion and Conclusion
Four questions were addressed in the team’s empirical study:

1. Is it possible to represent local text dynamics using Time Control?
2. Is it possible for Time Control to create locally coherent language?
3. Is it possible to represent global text dynamics using Time Control?
4. Is Time Control capable of producing long, cohesive documents?

For three tasks: discourse coherence, text-infilling, document structure imitating, and extended text production, they compared TC to domain-specific approaches and fine-tuning on GPT-2 across diverse text domains. Wiki section, TM-2, TicketTalk, and Recipe NLG were among the datasets used in the tests.


TC didn’t sacrifice short/mid-range language modeling performance as it improved performance on text infilling and discourse coherence tasks in the tests while preserving text structure for long text generation in terms of ordering (up to +40%) and text length consistency (up to +17%); this demonstrates the proposed method’s ability to generate more locally and globally coherent texts.

According to the team, TC may expand to other domains containing sequential data, such as movies or music, and support arbitrary bridge operations with unknown fixed start and endpoints.

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
