<h1>Detecting-Impasse-LAK2022</h1>

<h2>Description</h2>
<p>This is the repository for the following paper at the LAK conference 2022:</p> 
<p><em>Detecting Impasse During Collaborative Problem Solving with Multimodal Learning Analytics</em></p>

*[Paper Link](https://dl.acm.org/doi/abs/10.1145/3506860.3506865)*

<h3>Introduction</h3>
<p>Collaborative problem solving has numerous benefits for learners, such as improving higher-level reasoning and developing critical
thinking. While learners engage in collaborative activities, they often experience <em>impasse</em>, a potentially brief encounter with differing
opinions or insufficient ideas to progress. Impasses provide valuable opportunities for learners to critically discuss the problem and reevaluate
their existing knowledge. Yet, despite the increasing research efforts on developing multimodal modeling techniques to analyze
collaborative problem solving, there is limited research on detecting impasse in collaboration. This paper investigates multimodal
detection of impasse by analyzing 46 middle school learners’ collaborative dialogue—including speech and facial behaviors—during a
coding task. We found that the semantics and speaker information in the linguistic modality, the pitch variation in the audio modality,
and the facial muscle movements in the video modality are the most significant unimodal indicators of impasse. We also trained
several multimodal models and found that combining indicators from these three modalities provided the best impasse detection
performance. To the best of our knowledge, this work is the first to explore multimodal modeling of impasse during the collaborative
problem solving process. This line of research contributes to the development of real-time adaptive support for collaboration.</p>

<h3>Authors</h3>
Yingbo Ma, Mehmet Celepkolu, Kristy Elizabeth Boyer

<h3>Citation</h3>
<pre></pre>

<h2>Code Structure</h2>

Directories: 

(1) Features: this folder contains Python codes for feature extraction and post-processing.

(2) Prediction_Models: this folder contains Python codes for prediction models.

(3) For other data postprocessing details, please refer to my another personal repo: [Daily-Testing-Scripts](https://github.com/yingbo-ma/Daily-Research-Testing)

<h2>Prerequisites</h2>
<p>Basics</p>
<pre>
Python3 
CPU or NVIDIA GPU + CUDA CuDNN
</pre>
<p>Prerequisites for feature extraction</p>
<pre>
opencv 3.4.3
librosa 0.8.0
</pre>
<p>Prerequisites for model training</p>
<pre>
tensowflow-gpu 2.1.0
keras 2.3.1
</pre>

<p>Language-based Feature Extraction: TF-IDF, Word2Vec, Pre-trained BERT, Speaker-Aware BERT</p> 
<pre>
nltk v3.5
gensim v3.8.0
bert-for-tf2 v0.14.9
tensowflow-gpu v2.4.1
</pre>

<p>Video-based Feature Extraction: Eye Gaze, Head Pose, Facial AUs</p> 
<pre>
Feature extraction process was performed through command line arguments.
</pre>

<p>Prediction Models</p> 
<pre>
tensowflow-gpu v2.4.1
NVIDIA GPU + CUDA CuDNN
</pre>
