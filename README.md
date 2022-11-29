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

## Introduction

Existing generative models such as GAN (Goodfellow et al., 2014), VAE (Kingma & Welling, 2014) and Flow-based models (Dinh et al., 2017) suffer from the instability issue (Salimans et al., 2016), subjecting to mode collapse (Metz et al., 2017), and have to rely on surrogate objectives to approximate maximum likelihood training. Diffusion models (Ho et al., 2020; Nichol & Dhariwal, 2021) have circumvented several of these limitations and emerged as a new paradigm for generative models, theoretically underpinned by non-equilibrium thermodynamics (Sohl-Dickstein et al., 2015) and score-matching network (Song & Ermon, 2019). To date, the major breakthroughs are in domains using continuous signals, such as vision (Saharia et al., 2022a;b; Ramesh et al., 2022) and audio (Kong et al., 2020). However, extending continuous diffusion models to natural language remains an open challenge due to the inherently discrete nature of texts.

On the basis of unconditional generation in continuous space (illustrated in Figure 1(a)), existing efforts (Hoogeboom et al., 2021; Austin et al., 2021) start customizing diffusion models to text in discrete space on unconditional language modeling (i.e., free text generation). Diffusion-LM (Li et al., 2022) (as in Figure 1(b)) models texts in continuous space and proposes to use a extra-trained classifier as a guidance (i.e., the condition signal x) to impose subtle changes (usually complex, finegrained constraints) on generated sentences. Nonetheless, these models do not naturally generalized to conditional language modeling (i.e., the model assigns probabilities p(w|x) to sequences of words w given x). In the more general sequence-to-sequence (SEQ2SEQ) setting where the condition x is also a sequence of words, applying Diffusion-LM can be difficult. The reason is that classifiers are attributes-oriented, and we can not train hundreds-of-thousands classifiers to model the semantic meaning between conditions and generated sentences.

SEQ2SEQ is an essential setting in NLP that covers a wide range of important tasks such as openended sentence generation, dialogue, paraphrasing, and text style transfer. In this paper, we propose DIFFUSEQ (in Figure 1 (c)), a classifier-free diffusion model that supports SEQ2SEQ text generation tasks. By modeling the conditional probability of the target sentence w given context x using one single model, one advantage of DIFFUSEQ is that this paradigm allows a complete model to fit data distribution and utilize conditional guidance, rather than depending on a separate classifier.

To corroborate the effectiveness of our DIFFUSEQ, we conduct experiments on four SEQ2SEQ tasks. Compared to autoregressive (AR) and non-autoregressive (NAR) models, which suffer from the “degeneration” problem (Holtzman et al., 2019) and rely on decoding strategies, DIFFUSEQ can achieve considerable sentence-level diversity without sacrificing the quality (see § 4.2).

To sum up, we make a series of technical and conceptual contributions: (a) we are the first to deploy the diffusion model on SEQ2SEQ text generation, and our proposed DIFFUSEQ as a conditional language model is trained end-to-end in a classifier-free manner; (b) we establish a theoretical connection among AR, NAR and DIFFUSEQ models, and justify DIFFUSEQ as an extension of iterative-NAR models; (c) with strong empirical evidence, we demonstrate the great potential of diffusion models in complex conditional language generation tasks.

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
