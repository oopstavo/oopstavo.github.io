---
layout: true
title: Decision Support System
comments: true
tags:
  - UOFT
  - MIE1513
  - Machine Learning
categories:
  - CS
date: 2018-12-09 19:00:18
updated:
summary: Based on MIE1513 in University of Toronto.
permalink:
---

> Notes from **MIE1513 (2018 Fall)** in **University of Toronto**. Professor is **Scott Sanner**.

---
### **Topic 1: Information Retrieval**
1.A Introduction
**Information Retrieval** is finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers).
**Collection**: A set of documents.
**Goal**: Retrieve documents with information that is relevant to the user's information need and helps the user complete a task.

**Tracking Methods**
- Term-document incidence Matrices: 1 if document contains term, 0 otherwise.
- Inverted Index: for each term t, store a **sorted** (easy to merge) list of all documents that contain t.
- Positional Index (easy for phrase query): In the postings, store, for each term the position(s) in which tokens of it appear.

**Boolean Queries**: Exact match. The boolean retrieval model is being able to ask a query that is a Boolean expression. Many search system like Email, library catelog, Mac OS X Spotlight still use boolean.
**Good for expert users with precise understanding of their needs and the collection. Not good for the majority of users.**
**Problems with Boolean search:** feast or famine. Boolean queries often result in either too few (=0) or too many (1000s) results.

IR vs. Databases (Structured vs. Unstructured data):
- Structured data typically allows numerical range and exact match (for text) queries.
- Unstructured data allows keyword queries including operators.

1.B Rank retrieval
**Score**:
- Jaccard coefficient: It doesn’t consider term frequency (how many times a term occurs in a document). Rare terms in a collection are more informative than frequent terms. Jaccard doesn’t consider this information.
- tfidf score: $$ tf.idf_{t,d} = (1+log(tf_{t,d}))*log_{10}(\frac{N}{df_{t}}) \\ Score(q,d) = \sum_{t\in q\cap d}tf.idf_{t,d}$$
- cosine similarity: Rank documents according to cosine similarity with query.
    
**Bag of words model**: Vector representation doesn’t consider the ordering of words in a document.

1.C Evaluation
**Precision**: Fraction of retrieved docs in collection that are retrieved. 
$$ Precision P = \frac{tp}{(tp+fp)} $$
**Recall**: Fraction of relevant docs in collection that are retrieved.
$$ Recall R = \frac{tp}{(tp+fn)} $$
tp: relevant & retrieved.
tn: nonrelevant & not retrieved
fp: nonrelevant & retrieve
fn: relevant & not retrieveds


---
### **Topic 2: Machine Learning**
---
### **Topic 3: Recommender System**
---
### **Topic 4: Text Mining / Natual Language Processing**
#### ***1. NLP Text Processing Pipeline***
>- Document -> Sections and Paragraphs
- Paragraphs -> Sentences (sentence segmentation / extraction)
- Sentences -> Tokens
- Tokens -> Lemmas or Morphological Variants / Stems
- Tokens -> Part-of-speech (POS) Tags
- Tokens, POS Tags -> Phrase Chunks (Noun & Verb Phrases)
- Tokens, POS Tags -> Parse Trees

**Type**: an element of vocabulary. The term "type" refers to the number of distinct words in a text, corpus etc. V = vocabulary = set of types.
**Token**: an instance of type in running text. The term "token" refers to the total number of words in a text, corpus etc, regardless of how often they are repeated. N = number of tokens.
$$ |V| > O(N^\frac{1}{2}) $$
**Lemmatization**: reduce inflections or variant forms to base form. Have to find correct dictionary headword form.
**Stemming**: reduce terms to their stems in information retrieval. Stemming is crude chopping of affixes.
**Regular Expressions**: A formal language for specifying text strings. 
> In NLP we are always dealing with these kinds of errors: false positives, false negatives. 
- Increasing accuracy or precision (minimizing false positives).
- Increasing coverage or recall (minimizing false negatives).

#### ***2. Advanced NLP Processing***
**Part-of-speech tagging**: a simple but useful form of linguistic analysis. The POS tagging problem is to determine the POS tag for a particular instance of a word.

**Phrase Chunking**: find all non-recursive noun phrases (NPs) and verb phrases (VPs) in a sentence.

**Statistical Natual Language Parsing**
Two views of linguistic structure:
- Constituency (phrase structure): Phrase structure organizes words into nested constituents.
  - **constituent**: Constituent behaves as unit that can appear in different places (John talked [to the children] [about drugs].John talked [about drugs] [to the children].)
- Dependency structure: Dependency structure shows which words depend on (modify or are arguments of) which other words. Can derive dependency tree from parse tree

#### ***3. Sentiment Analysis***
**Sentiment analysis is the detection of attitudes.**
**Scherer Typology of Affective States**:
- Emotion: brief organically synchronized ... evaluation of a major event: angry,sad,joyful,fearful,ashamed,proud,elated
- Mood: diffuse non-caused low-intensity long-duration change in subjective feeling: cheerful, gloomy, irritable, listless, depressed, buoyant
- Interpersonal stances: affective stance toward another person in a specific interaction: friendly,flirtatious,distant,cold,warm,supportive,contemptuous
- Attitudes: enduring, affectively colored beliefs, dispositions towards objects or persons: liking, loving, hating, valuing, desiring
- Personality traits: stable personality dispositions and typical behavior tendencies: nervous,anxious,reckless,morose,hostile,jealous

**Analyse aspect**:
- Holder (source) of attitude 
- Target (aspect) of attitude 
- Type of attitude
  - From a set of types: Like,love,hate,value,desire,etc.
  - Or (more commonly) simple weighted polarity: positive,negative,neutral,togetherwithstrength
- Text containing the attitude: Sentence or entire document

**Baseline Algorithm**
- Tokenization
- Feature extraction
- Classification using different classifiers (need to train a classifier per domain)
  - Naive Bayes
  - MaxEnt
  - SVM (preform better than others)
  
**Word occurrence may matter more than word frequency**

**Sentiment Lexicons**
- The General Inquirer
- LIWC (Linguistic Inquiry and Word Count)
- MPQA Subjectivity Cues Lexicon
- Bing Liu Opinion Lexicon
- SentiWordNet (do not use)

**Analyzing the polarity of each word**:
- Scaled likelihood: $$ \frac{P(w|c)}{P(w)} $$

**Semi-supervised learning of lexicons**: Use a small amount of information to bootstrap a lexicon. (Adjectives conjoined by “and” have same polarity, adjectivesconjoinedby“but”donot)
- Label seed set of 1336 adjectives
- Expand seed set to conjoined adjectives
- Supervised classifier assigns “polarity similarity” to each word pair.
- Clustering for partitioning the graph into two. 

**Turney Algorithm**:
- Extract a phrasal lexicon from reviews
- Learn polarity of each phrase
  - Positive phrases co-occur more with “excellent”
  - Negative phrases co-occur more with “poor”
  - Calculate the PMI of each words with "excellent" or "poor"
- Rate a review by the average polarity of its phrases

**Mutual Information**: between 2 random variables X and Y 
$$ I(X,Y) = P(x,y)log_{2}\frac{P(x,y)}{P(x)P(y)s} $$
**Pointwise Mutual Information**: How much more do events x and y co-occur than if they were independent?
$$ PMI(x,y) = log_{2}\frac{P(x,y)}{P(x)P(ys)} $$
P(word) estimated by hits(word)/N
P(word1,word2) by hits(word1 NEAR word2)/N^2
Polarity( phrase) = PMI( phrase, "excellent") - PMI( phrase, "poor")

**Finding aspect/attribute/target of sentiment**
- Frequent phrases + rules
  - Find all highly frequent phrases across reviews (“fish tacos”)
  - Filter by rules like “occurs right after sentiment word”
- The aspect name may not be in the sentence
- Supervised classification
  - Hand-label a small corpus of restaurant review sentences with aspect
  - Train a classifier to assign an aspect to a sentence

Hierarchical Clustering: Build a tree-based hierarchical taxonomy (dendrogram) from a set of documents.
- One approach: recursive application of a partitional clustering algorithm. Clustering obtained by cutting the dendrogram at a desired level: each connected component forms a cluster.

Hierarchical Agglomerative Clustering (HAC):
- Starts with each doc in a separate cluster, then repeatedly joins the closest pair of clusters, until there is only one cluster.
- Closest pair of clusters:
  - Single-link(distance of nearest points): Similarity of the most cosine-similar (single-link). Can result in “straggly” (long and thin) clusters due to chaining effect. For most applications, these are undesirable.
  - Complete-link(distance of furthest points): Similarity of the “furthest” points, the least cosine-similar. Makes “tighter,” spherical clusters that are typically preferable. Notice that this dendrogram is much more balanced than the single-link one. We can create a 2-cluster clustering with two clusters of about the same size.
  - Centroid: Clusters whose centroids (centers of gravity) are the most cosine-similar. The similarity of two clusters is the average intersimilarity – the average similarity of documents from the first cluster with documents from the second cluster.
  - Average-link: Average cosine between pairs of elements


---
### **Topic 5: Data Science and Visualization**

**Beware: Spurious Correlations**
---
### **Topic 6: Social Network Analyse**
#### ***1. Examples of Problems in Social Networks***
#### 1.A. Small-world phenomena & Link Prediction
> **The small-world phenomenon** -- the principle that we are all linked by short chains of acquaintances, or "six degrees of separation" -- is a fundamental issue in social networks; it is a basic statement about the abundance of short paths in a graph whose nodes are people, with links joining pairs who know one another.
-- [Math Awareness Month - April 2004](http://www.mathaware.org/mam/04/essays/smallworld.html)  

<!--  -->
> ***Link Prediction***: Given a snapshot of a social network, can we infer which new interactons among its members are likely to occur in the near future?

- **Measurements**: "proximity" and "similarity" between two unconnected nodes.
- **Applicaton Domains**: social networks, friend recommendaton; product / webpage recommendaton; predictng academic collaboratons; predictng merger and acquisitons ...
- **Performance Measurements**: use future held-out data, conversion rate.

#### 1.B. Tracking Memes - What goes Viral?

#### 1.C. Community Detection 
>**Why detect communites?**:

- Simplifed network structure and “big picture”
- Explain actons and positons
- Better predictons

>**What defines a community?**

- Interacton
- Profle
- Dynamics

#### ***2. Data Representation***
> #### 2.A. Adjacency Matrices  

Representng edges (who is adjacent to whom) as a matrix
$$ A_{ij} = \begin{cases} 1, \text{ if node i has an edge to node j } \\0,\text{ if node i does not have an edge to j} \end{cases} $$
$$A_{ii} = 0, \text{unless the network has self-loops} $$
$$A_{ij} = A_{ji}, \text{ if the networkif undircted, or if i and j share a reciprocated edge}$$

> #### 2.B. Edge List  

```
# edges
- 2,3
- 2,4
- 3,2
```

> #### 2.C. Adjacency List  

- Adjacency lists are easier to work with if network is **large** or **sparse**.
- Adjacency lists quickly retrieve all neighbors for a node.
```
# node: list of adjacency nodes.
1:
2: 3, 4
3: 2, 4
```

#### ***3. Computing Metrics***
> #### 3.A. Path and distance

- Path: a walk (i1,i2,... ik) with each node ij distnct 
- Cycle: a walk where i1 = ik
- Geodesic: a shortest path between two nodes

> #### 3.B. In/Out degree

- Indegree: how many directed edges (arcs) are incident on a node
- Outdegree: how many directed edges (arcs) originate at a node 
- Degree (in or out): number of edges incident on a node

Calculation in Adjacency Matrix:
- Outdegress: summing the number of nonzero entries in the ith row.
$$\sum_{j=1}^n A_{ij}$$
- Indegree: summing the number of nonzero entries in the jth column.
$$\sum_{i=1}^n A_{ij}$$

> #### 3.C. Centrality

**Betweenness**: centrality capturing brokerage. (would have to go through you in order to intuition: how many pairs of individuals reach one another in the minimum number of hops?)
> **Definition**: 
$$ C_B(i)=\sum_{j<k} g_{jk}(i)/g_{jk} \\
\text{where } g_{jk} = \text{the number of shortest paths connecting jk} \\
g_{jk}(i) = \text{the number that vertex i is on. } $$

**Normalization**: $$ C'_{B}(i) = C_{B}(i)/[(n-1)(n-2)/2] \text{ (divide by number of pairs of vertices excluding the vertex itself)} $$
**Betweenness Clustering** (also known as [Girvan–Newman algorithm](https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm))
*Edge Betweenness*:
- The "edge betweenness" of an edge as the number of shortest paths between pairs of nodes that run along it.

*Algorithm Steps*:
- The betweenness of all existing edges in the network is calculated first.
- The edge with the highest betweenness is removed while betweenness of any edge larger than threshold.
- The betweenness of all edges affected by the removal is recalculated.
- Steps 2 and 3 are repeated until no edges remain.

*Problems with this algorithm*:
- very expensive: all pairs shortest path – O(N^3)
- may need to repeat up to N times
- does not scale to more than a few hundred nodes, even with the fastest algorithms

**PageRank Centrality**: A technique for estimating page quality based on web link graph.
- Each page i is given a rank Xi
- **Goal**: Assign the Xi such that the rank of each page is governed by the ranks of the pages linking to it: $$ x_{i} = \sum_{j\in B_{i}} \frac{1}{N_{j}}  x_{j} \\ x_{i}:\text{ Rank of page i} \\ B_{i}: \text{ Every page j that links to i} \\ N_{j}: \text{ Number of links out from page j} \\ x_{j}: \text{ Rank of page j}$$

*PageRank Sinks*: If a webpage doesn't have links to other webpages, after iterations, PageRank will be 0 for all webpages.
    
*PageRank Hogs*: If a webpage has a link to itself, after iterations, PageRank of this page will the sum of all other webpages.
*Improved PageRank*: 
- Remove out-degree 0 nodes (or consider them to refer back to referrer)
- Add decay factor d to deal with sinks
$$ PageRank(p) = (1-d)+d\sum_{b\in B_{p}}\frac{1}{N(b)} PageRank(b) \text{ Typical value of d is 0.85}$$
*PageRank is useful, but vulnerable to black-hat SEO.*

#### ***4. Link Prediction***
Given G[t0,t'0] a graph on edges up to time t'0 output a ranked list L of links (not in G[t0,t'0]) that are predicted to appear in G[t1,t'1]
**Evaluation**:
- n = |Enew|: number of new edges that appear during the test period [t1,t'1]
- Take top n elements of L and count correct edges.

**Link Prediction via Proximity**
- For each pair of nodes (x,y) compute score c(x,y) (number of common neighbors c(x,y) of x and y)
- Sort pairs (x,y) by the decreasing score c(x,y)
- Predict top n pairs as new links and see which of these links actually appear in G[t1,t'1]

**Different scoring functions c(x,y)** s
- Graph distance: (negated) shortest path length
- Common neighbors: $$ |\Gamma(x)\cap\Gamma(y)| (\Gamma(x) \text{ is neighbors of node x})$$
- Jaccard's coefficient: $$ \frac{|\Gamma(x)\cap\Gamma(y)|}{|\Gamma(x)\cup\Gamma(y)|} $$
- Adamic/Adar: $$ \sum_{Z\in \Gamma(x)\cap\Gamma(y)}\frac{1}{log|\Gamma(Z)|} $$ 
- Preferential attachment: $$ |\Gamma(x)|*|\Gamma(y)| $$
- PageRank: $$ r_{x}(y) + r_{y}(x) $$
  * rx(y) stationary distribution weight of y under the random walk:
    - with prob. 0.15, jump to x
    - with prob. 0.85, go to random neighbor of current node.









---
