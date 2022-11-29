# <img src="img/logo.jpg" width="8%" alt="" align=center /> DiffuSeq

<div align="center">
|                              TITLE                              |     AUTHOR     |     DATE    |  CATEGORY  |
| --------------------------------------------------------------- |:--------------:| :----------:|:----------:|
| Sequence To Sequence Text Generation With Diffusion Models      | Shubham Talbar |  2022-10-25 |     NLP    |
</div>
  
<p align = "left">
The following blog is an attempt to simplify and understand the use of Diffusion Models in complex conditional language generation tasks. This article is based on the paper DIFFUSEQ: Sequence To Sequence Text Generation With Diffusion Models
</p>
  
<!-- more -->

Paper: <https://arxiv.org/pdf/2210.08933.pdf>

GitHub: <https://github.com/Shark-NLP/DiffuSeq>

Official Codebase for [*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*](https://arxiv.org/abs/2210.08933).

&nbsp;

## Introduction

Recently, diffusion models have emerged as a new paradigm for generative models. Despite the success in domains using continuous signals such as vision and audio, adapting diffusion models to natural language is difficult due to the discrete nature of text. We tackle this challenge by proposing DiffuSeq: a diffusion model designed for sequence-to-sequence (Seq2Seq) text generation tasks. Upon extensive evaluation over a wide range of Seq2Seq tasks, we find DiffuSeq achieving comparable or even better performance than six established baselines, including a state-of-the-art model that is based on pre-trained language models. Apart from quality, an intriguing property of DiffuSeq is its high diversity during generation, which is desired in many Seq2Seq tasks. We further include a theoretical analysis revealing the connection between DiffuSeq and autoregressive/non-autoregressive models. Bringing together theoretical analysis and empirical evidence, we demonstrate the great potential of diffusion models in complex conditional language generation tasks.

&nbsp;

<p align = "center">
<img src="img/diffuseq-process.png" width="95%" alt="" align=center />
</p>

<p align = "center">
<strong>The diffusion process of our conditional diffusion language model DiffuSeq</strong>
</p>

&nbsp;

## Significant Contributions
The team’s significant contributions are summarised as follows:

1. Time Control is a language model derived by the team that explicitly represents latent structure using Brownian bridge dynamics acquired with a new contrastive aim.
2. Compared to task-specific approaches, the team showed that Time Control creates more or equally coherent text on tasks such as text infilling and forced lengthy text production across various text domains.
3. By evaluating discourse coherence with human studies, The team demonstrates that their latent representations capture text dynamics competitively.
4. The relevance of the contrastive aim, enforcing Brownian bridge dynamics, and explicitly modeling latent dynamics are all emphasized in their technique.

## Method

The proposed TC approach learns a latent space with smooth temporal dynamics for modeling and creating coherent text. The researchers devised a unique contrastive goal for learning a latent space with Brownian bridge dynamics and then utilized this latent space to create text that keeps local coherence while displaying better global coherence.

![image3](./image3.jpeg)

The TC text generation pipeline uses the Brownian bridge process to plan a latent trajectory with a start and finish, then conditionally creates sentences that follow this latent plan.

The intuition is simple: The bridge imposes that a positive triplet (eg. three in-order sentences on Boston) makes up a smooth trajectory. A negative triplet should not construct a smooth trajectory (switching middle sentences with one on New York).

After training the encoder, GPT2 is finetuned to decode from past context and the encoded latent plan. At inference, a latent plan is generated by sampling from the bridge and conditionally generating each sentence using the latent plan.


![image1](./image1.jpeg)

## Discussion and Conclusion
Four questions were addressed in the team’s empirical study:

1. Is it possible to represent local text dynamics using Time Control?
2. Is it possible for Time Control to create locally coherent language?
3. Is it possible to represent global text dynamics using Time Control?
4. Is Time Control capable of producing long, cohesive documents?

For three tasks: discourse coherence, text-infilling, document structure imitating, and extended text production, they compared TC to domain-specific approaches and fine-tuning on GPT-2 across diverse text domains. Wiki section, TM-2, TicketTalk, and Recipe NLG were among the datasets used in the tests.

![image5](./image5.png)

![image6](./image6.png)

TC didn’t sacrifice short/mid-range language modeling performance as it improved performance on text infilling and discourse coherence tasks in the tests while preserving text structure for long text generation in terms of ordering (up to +40%) and text length consistency (up to +17%); this demonstrates the proposed method’s ability to generate more locally and globally coherent texts.

![image4](./image4.png)



According to the team, TC may expand to other domains containing sequential data, such as movies or music, and support arbitrary bridge operations with unknown fixed start and endpoints.

## Reference 
- Wang, R. E., Durmus, E., Goodman, N., & Hashimoto, T. (2022). Language modeling via stochastic processes. arXiv preprint arXiv:2203.11370.
