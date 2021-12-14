# Neural Machine Translation

Our project focused on low resource language translation by implementing supervised systems which leverage transfer learning and pretrained models. Our models include both a baseline Seq2Seq model, as well as a pivot learning based model using Spanish (ES) as a pivot language. Our results found that the pivot based model performed similarly to the baseline approach, although this came at significant computational increases.


## 1 Introduction

The spark for this project was a [paper](https://arxiv.org/pdf/1909.09524.pdf) focused on the idea of transfer learning within neural language translation. We were interested in seeing how well this technique would perform in the low resource setting. Intuitively, we assumed that being able to leverge a large parallel corpus with a pivot language would allow for a more accurate model compared to using an extremely limited source to target corpus.

Our project is focused on the translation of Catalan (ca)  Italian (it). For our baseline system, we employed an Encoder-Decoder architecture with an attention mechanism, similar to *Attention is all you need*  (Vaswani et al.,
2017). However, these models require significant amounts of training data, and limited data (such as is the case for this shared task) can negatively impact results that can be achieved with these models (Koehn and Knowles, 2017). In order to circumvent the limited amount of aligned training data between these language pairs, we leveraged a pivot learning based approach using Spanish (es) as our pivot language for our target. We chose to use Spanish as our pivot language for two main reasons. First, the Catalan language is primarily spoken within regions in Spain. Many of these regions (such as Barcelona, within Catalonia) recognize both Catalan and Spanish as official languages, and fluency in both is prominent among speakers in these regions. As a result of both being recognized as official languages, as well as the prominence of both in the same area, there is a large amount of parallel data for Catalan and Spanish. Another reason is that the historical proximity of these languages has led to significant amounts of borrowing between these languages. As a result, many aspects of the language such as syntactical elements will be preserved when translating into Spanish and therefore the model will not have to learn them. The prevalence of borrowing also opens up the opportunity to hard-code common aspects of these translations, in order to further decrease the overhead of what the model has to learn. 

Because Spanish is one of the most commonly spoken languages in the world (and the most common Romance language), there are also large amounts of training data available from Spanish to our target language. This allows us to build a competent model to translate from our pivot language to our target, as well as opening up the feasibility of using pretrained models for this step of translation. 

In addition to standard pivot learning, we employ architecture tweaks in order to boost efficiency and accuracy. These include step-wise transfer learning, as well as the use of large pretrained models.

## 2 Related Work

Transfer learning has been a popular approach for the low resource setting and can be found used by many teams for the WMT shared [task](http://www.statmt.org/wmt20/unsup_and_very_low_res/). In this section we will discuss a few papers which helped guide our work and direction.

*Pivot-based Transfer Learning for Neural Machine Translation between Non-English Languages* (Kim et al., 2019): This paper served as a guide for our transfer learning approachs and provided us with the outline for implementation for our various techniques. In their paper, the authors further expand on the step-wise pretraining approach through the use of a pivot adapter and a cross-lingual encoder. A step-wise approach coupled with these modifications sees a noticeable improvement from basic transfer learning or base step-wise pretraining. Unlike our results, they saw that step-wise pretraining performed the best and transfer learning performed better than the baseline approach. Interestingly enough, they show that these techniques perform well even under a zero-shot scenario (no fine-tuning).

*Trivial Transfer Learning for Low-Resource Neural Machine Translation* (Kocmi and Bojar 2018): This paper also implements a transfer learning approach for low resource senarios. However, they explore how well the model performs under various very low resource senarios. They do this by varying the training set from 10k to 800k sentence pairs. As the size of the training set increased, the baseline seq2seq approach improved significantly faster than the transfer learning approch. However, in the extreme low resource scenario (10k sentence pairs), the transfer learning model performed approximately 6 times better than the basline model posting a BLEU score of 12.46 compared to the basline's 1.95.

## 3 Data

Since our techniques are focused on using a pivot language, we needed in total three pairs of parallel corpora. Using the provided list of allowed sources, we use WikiMatrix Schwenk et al., 2019) for ca-it, ParaCrawl (Morishita et al., 2020)  for ca-es, and MultiCCAligned (El-Kishky et al., 2020) for es-it. For our experiments, we focused on Catalan to Italian using Spanish as a pivot. Table 1 summarizes the datasets used before and after preprocessing.

Table 1:

| Dataset | Sentence Count | Post-Preprocessing |
| --------------- | --------------- | --------------- |
| ca-it | 31M | 30M |
| es-it | 28M | 27M |
| ca-it | 316K | 308K |

### 3.1 Preprocessing

We know from previous works that neural networks based on encoder-decoder transformers tend to perform poor translations on longer sentences (Koehn and Knowles, 2017). To help mitigate this issue, we remove sentences containing 60 or more tokens from our corpora. To tokenize our data, we use SentencePiece (Kudo and Richardson, 2018). The final sentence counts can be seen in Table 1.

For each parallel data, we generate our own test and validation set (90:5:5).

## 4 Approach

In this section, we provide details on our particular experiments and implementation of models for translation between Catalan and Italian.

### 4.1 Baseline

To properly compare the influence of our transfer learning approaches, we implemented a direct source to target model. This was a standard NMT model trained on the ca-it parallel corpus. This is the simplest approach and takes considerably less training time and resources when compared to the transfer learning methods.

The other baseline we used was passing a sentence through multiple NMTs. This involved an NMT from source $\rightarrow$ pivot and another from pivot $\rightarrow$ target. The final translation would pass the test sentence through both models to get a final translation.

### 4.2 Pivot-based Transfer Learning

For this method we apply transfer learning using Spanish as a pivot language. We can summarize our process into the following steps:

1. Train NMT from source &#8594; pivot corresponding to ca &#8594; es using our ca-es parallel corpus.
2. Train NMT from pivot &#8594; target corresponding to es &#8594; it using our es-it parallel corpus.
3. Copy the parameters from the source &#8594; pivot encoder and the parameters from the pivot &#8594; target decoder into a new model. Train one last time using the source &#8594; target parallel corpus (ca-it).

### 4.3 Step-wise Transfer Learning

One issue with the above pivot learning process is that our final model's encoder is trained to be connected to a ca &#8594; es decoder, while it's actually connected to a es &#8594; it decoder. Similarly, this es &#8594; it decoder was trained to expect its input to come from an es &#8594; it encoder. These discrepancies can produce a mismatch on how the intermediary vector stores meaning, and can lead to mal-formed translations.

In order to remedy this, we need a way for these separated encoder and decoders to become aware of each other under training. In order to do so, we will employ a sequential training process in which we carry over the parameters of the final encoder when training the final decoder rather than training both in isolation. 

This approach is similar to the pivot-based method with a small change to how the pivot $\rightarrow$ target model is trained. Here are the steps used:

1. Train NMT from source &#8594; pivot corresponding to ca &#8594; es using our ca-es parallel corpus.
2. Copy the parameters from the source &#8594; pivot encoder, freeze the encoder, and train using our it-es parallel data.
3. Unfreeze the encoder, and train using the source &#8594; target parallel corpus (ca-it). 

### 4.4 Employing Pretrained Models

Since the shared task allows for the use of pretrained models, we explored using one similar to our second baseline. In this experiment, we used Facebooks mBart-large-50 many to many model for multilingual translation tang2020multilingual. While this model does not support translation from Catalan to Italian, it does have Spanish to Italian. Therefore, we first pass our test data through our ca &#8594; es model and then through the mBart model set to translate between Spanish and Italian. We use this approach in order to leverage the benefits of large compute used for this pretrained model, as well as the efficiency of reducing the need to re-train models.

One potential issue with this is that the use of this complete model necessitates that we affix it to our ca &#8594; es model which can result in a telephoning effect, as well as doubling the decoding time. Ideally we would be able to take only the decoder from this model, and leverage the strengths of this pretrained model if we can encode the information from the source language in the format appropriate for this decoder.

This also opens up the question: What if we could make our ca &#8594; es encoder aware of the parameters of this pretrained model under training? This would limit the telephoning effect and decoding time while also leveraging the strengths of using a pretrained model. 

## 5 Implementation 

We used the 6-layer base Transformer architecture (Vaswani et al., 2017) for all of our NMT models. For all
model training setups, we learned byte pair encoding (BPE) (Sennrich et al., 2016) for each language individually with 32k merge operations. We utilize SentencePiece  (Kudo and Richardson, 2018) accomplish this. The model training was done with the OpenNMT toolkit (Klein et al., 2018). Batch size was set to 3,072 tokens. For our scheduler, we used NoamDecay described by (Vaswani et al., 2017) with an initial learning rate of 0.0001. For our optimization, we used LazyAdam  (Kingma and
Ba, 2017) with a warm up of 8000 steps. We trained for a total of 25k steps for each of the techniques provided in Table 2. This includes fine-tuning the pivot-based and step-wise approach for 25k steps for the source-pivot and pivot-target models. 

## 6 Results

The results of our systems on the ca-it test set are presented in Table 2.

For the baselines, we see the direct source &#8594; target model surpass that of the telephoning approach, posting a +10.9\% increase in BLEU score.

Similar to the telephoning method but using Facebook's mBart large 50 many to many NMT model instead of the standard NMT between Spanish &#8594; Italian, we see a similar performance in terms of both TER and BLEU. We saw that it slightly outperformed the baseline scoring by +1.5\% in BLEU. 

Looking at the transfer learning based approaches, we see the standard pivot-based approach performs the best out of the two with a BLEU score of 23.0, a +51.3\% increase over the step-wise approach and +8.7% over the pretrained telephoning. However, it sees a slightly worse BLEU score when compared to the direct NMT baseline model, seeing a -1.7% in BLEU.

In conclusion, we see the Direct source &#8594; target NMT performed the best with the pivot-based approach showing similar results. We found the step-wise transfer method produced the lowest scores.\

Table 2:

| Model | BLEU | TER |
| --------------- | --------------- | --------------- |
| Direct | 23.4 | 0.72 |
| Telephoning | 21.1 | 0.75 |
| Pivot-based | 23.0 | 0.74 |
| Step-wise | 15.2 | 0.79 |
| Pretrained | 21.4 | 0.75 |

## 7 Discussion

Ultimately, the direct baseline approach had the best performance in our translation between Catalan and Italian, slightly outperforming the Pivot-based transfer learning approach. 

Here are a few insights we gained:


### 7.1 How transfer learning reduces the impact of telephoning

The main issue we see when creating the pivot-based approach is that in the final model, the Catalan encoder is pretrained to output and be used by a Spanish decoder and similarly, the Italian decoder is pretrained to use outputs from a Spanish encoder. This mismatch is addressed in our step-wise model.

By freezing the encoder during our Spanish &#8594; Italian training, we ensure the encoder is still modeling Catalan. However, in our results, these did not manifest themselves into an improvement in either BLEU or TER. Rather, we saw a worse performance using this technique. This is a rather unexpected result when compared to other reports using similar implementations (Kim et al., 2019). Likely, this is due to a fault in our code. It is rather involved to freeze an entire encoder using OpenNMT especially the attention layers. 

### 7.2 Telephoning leads to compound error

A basic result which we find after our experimentation is that stringing multiple trained NMTs together tends to be worse than simply training a direct model. Even if we have a large abundance of data to and from a pivot language we see a compounding effect when we pass a sentence through multiple translations. 

We find that the final translation can be fragmented. This can be in large part due to infrequent words in the test sentence leading to an unknown token from the source to pivot model. The error we see from the first translation is only worsened by passing it through our second model. We therefore see a drop in the BLEU score.

### 8 References

Ahmed El-Kishky, Vishrav Chaudhary, Francisco Guzmán, and Philipp Koehn. 2020. CCAligned: A massive collection of cross-lingual 
    web-document pairs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020).
    
Yunsu Kim, Petre Petrov, Pavel Petrushkov, Shahram Khadivi, and Hermann Ney. 2019. Pivot-based transfer learning for neural machine translation between non-english languages. 

Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization.

Guillaume Klein, Yoon Kim, Yuntian Deng, Vincent Nguyen, Jean Senellart, and Alexander M. Rush. 2018. Opennmt: Neural machine translation toolkit. 

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation, pages 28–39, Vancouver. Association for Computational Linguistics.

Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. 

Makoto Morishita, Jun Suzuki, and Masaaki Nagata. 2020. JParaCrawl: A large scale web-based English Japanese parallel corpus. In Proceedings of The 12th Language Resources and Evaluation Conference, pages 3603–3609, Marseille, France. European Language Resources Association.

Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzmán. 2019. Wikimatrix: Mining 135m parallel sentences in 1620 language pairs from wikipedia.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units.

Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. 2020. Multilingual translation with extensible multilingual pretraining and finetuning. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need.

Kocmi, Tom and Bojar, Ond. 2018. Trivial Transfer Learning for Low-Resource Neural Machine Translation. In Proceedings of the Third Conference on Machine Translation: Research Papers.
