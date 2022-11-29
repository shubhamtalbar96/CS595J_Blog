# <img src="img/logo.jpg" width="8%" alt="" align=center /> DiffuSeq

|                              TITLE                              |     AUTHOR     |     DATE    |  CATEGORY  |
| --------------------------------------------------------------- |:--------------:| :----------:|:----------:|
| Sequence To Sequence Text Generation With Diffusion Models      | Shubham Talbar |  2022-10-25 |     NLP    |


The following blog is an attempt to simplify and understand the use of Diffusion Models in complex conditional language generation tasks. 

This article is based on the paper: DIFFUSEQ: SEQUENCE TO SEQUENCE TEXT GENERATION WITH DIFFUSION MODELS 

<!-- more -->

Paper: <https://arxiv.org/pdf/2210.08933.pdf>

GitHub: <https://github.com/Shark-NLP/DiffuSeq>

Official Codebase for [*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*](https://arxiv.org/abs/2210.08933).


## Introduction

Writing a few lines is an easy chore for most individuals, but even seasoned authors frequently run into difficulties when trying to construct their second chapter. A similar problem plagues today’s large-scaled pretrained language models, such as GPT-2, which excel at short text production but degrade into incoherence when used for lengthier texts. The incapacity of such models to plan or reflect long-range dynamics might be blamed for the failure to evolve texts from beginning to conclusion correctly.

![image2](./image2.png)

Prior work has explored remedies for this failure mode by using planning-based methods or implicitly learning text dynamics. However, these methods manually specify the text dynamics or sacrifice quality in long-horizon generation.


To address these challenges, a Stanford University research team introduced Time Control (TC), a language model that implicitly plans using a latent stochastic process and seeks to generate sentences that follow this secret plan. Human assessors scored the outputs 28.6 percent higher than baseline approaches, indicating that the unique strategy enhances performance on long text production.




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
