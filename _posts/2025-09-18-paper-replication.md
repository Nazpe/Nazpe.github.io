---
layout: post
title: Exploring the Political Agenda of the EU Parliament (with NMF)
subtitle: A paper replication 
gh-repo: Nazpe/Non-Negative-Matrix-Factorization
gh-badge: [star, follow]
tags: [NMF]
comments: true
mathjax: true
author: Nuno Pedrosa
---

A analysis and simulation of "Exploring the Political Agenda of the European Parliament Using a Dynamic Topic Modeling Approach" by Greene and Cross ["Exploring the Political Agenda of the European Parliament Using a Dynamic Topic Modeling Approach" by Greene and Cross](https://www.cambridge.org/core/journals/political-analysis/article/abs/exploring-the-political-agenda-of-the-european-parliament-using-a-dynamic-topic-modeling-approach/BBC7751778E4542C7C6C69E6BF954E4B).

The sheer volume of legislative speeches and documents produced by bodies like the European Parliament makes it impossible for humans to manually identify overarching themes and their evolution. This is where computational linguistics and machine learning come in. The paper here replicated, introduced a two-layer NMF method to tackle this challenge.

The goal was to replicate their core methodology: leveraging NMF to extract latent topics from a corpus of news articles and evaluate its effectiveness, particularly in comparison to LDA. Using NLP and ML methods with python libraries such as sklearn, nltk and gensim.

### Data Preparation

For this replication exercise, the sample corpus consisted of 1,324 news articles., divided into three monthly windows. The raw text needed significant cleaning and structuring before any modeling could begin.

First, we import the data with a simple script to read all text files into the processing pipeline.

```python
# expand pandas df column display width to enable easy inspection
pd.set_option('max_colwidth', 150)

# read the textfiles to a dataframe
dir_path = 'sample' # folder path
files = [] # list to store files

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(os.path.join(dir_path, path))
    else:
        subpath = os.path.join(dir_path, path)
        for path2 in os.listdir(subpath):
            if os.path.isfile(os.path.join(subpath, path2)):
                files.append(os.path.join(subpath, path2))
```

#### Tokenizing

The next crucial step was tokenization, which involves breaking down the text into individual words or "tokens." During this process, normalization was also performed:
*   Converting all text to lowercase.
*   Removing punctuation, numbers, and short words (length less than 3).
*   Eliminating common English stopwords (e.g., "the," "is," "a") that carry little semantic meaning for topic identification.
*   Counting term frequency within each document.

```python
text_tokens = dict()
for filename in files:
    with open(filename, 'rb') as f:
        lines = f.readlines()
        text_tokens[filename] = dict()
        
        for line in lines:
            for token in re.split('\W+', str(line)):
                token = token.lower()
                if len(token) > 3 and not token.isnumeric() and not token.lower() in stopwords.words('english'):
                    text_tokens[filename][token] = text_tokens[filename].get(token, 0) + 1
```

#### Lemmatizing

To ensure that different grammatical forms of a word (e.g., "walked," "walking," "walks") are treated as the same base word ("walk"), lemmatization was applied. This reduces sparsity in our data and helps to group related concepts. This specifically focused on lemmatizing nouns.

```python
wordnet_lemmatizer = WordNetLemmatizer()   # stored function to lemmatize each word
is_noun = lambda pos: pos[:2] == 'NN'

nouns = dict()
for filename, tokens in text_tokens.items():
    if filename not in nouns:
        nouns[filename] = dict()

    for (word, pos) in pos_tag(list(tokens.keys())):
        if is_noun(pos):
            nouns[filename][wordnet_lemmatizer.lemmatize(word)] = nouns[filename].get(wordnet_lemmatizer.lemmatize(word), 0) + text_tokens[filename][word]
```

#### Building the Document-Term Matrix (V)

The cleaned and lemmatized data was then transformed into a document-term matrix, **V**. In this matrix, each row represents a document (news article), and each column represents a unique term. The values initially represent the term frequency (how many times a term appears in a document).

```python
dictvectorizer = DictVectorizer(sparse=False)
a = dictvectorizer.fit_transform(list(nouns.values()))
# Resulting shape: (1324 documents, 11236 unique terms)
print(a.shape)

token_list = dictvectorizer.get_feature_names()
```

#### TF-IDF Weighting

Raw term frequencies can be misleading, as very common words might appear frequently in all documents without being particularly distinctive. To address this, we applied Term Frequency-Inverse Document Frequency (TF-IDF) weighting. TF-IDF gives higher weight to terms that are frequent in a specific document but rare across the entire corpus, effectively highlighting more meaningful words for topic identification. This also helps to produce diverse but semantically coherent topics.

```python
for column_idx in range(len(token_list)):
    idf = math.log(len(a[:, column_idx])/len([x for x in a[:, column_idx] if x != 0]), 10)

    for element_idx in range(len(files)):
        if a[element_idx,column_idx] != 0:
            a[element_idx,column_idx] = (math.log(a[element_idx,column_idx], 10) + 1) * idf
```

### Finding the Optimal Number of Topics (k) with TC-W2V

A critical decision in topic modelling is choosing the number of topics, `k`. Too few topics might merge distinct themes, while too many can create overly specific and redundant topics. Here Topic Coherence via Word2Vec (TC-W2V) was used, a method that evaluates the semantic relatedness of the top terms within a topic. A higher coherence score indicates a more semantically meaningful topic. Using `t = 10` terms per topic, as suggested in the original paper.

Iterating through a range of `k` values (10 to 25) and calculating the model coherence for each.

```python
max_model_coherence = 0
res_k = 0
t = 10 # number of terms per topic

for k in range(10,26):
    nmf_model = NMF(k, random_state=1) 
    nmf_model.fit_transform(a)

    vocabulary = [[token_list[x[1]] for x in sorted(zip(topic,range(len(topic))), reverse = True)[:t]] for topic in nmf_model.components_]
    model = Word2Vec(sentences = vocabulary, vector_size = 200, window = 5, hs = 1, negative = 0, min_count = 1)
    
    model_score = []
    for topic in vocabulary:
        topic_score = []
        for w1 in topic:
            for w2 in topic:
                if w2 > w1:
                    word_score = cosine_similarity(model.wv[w2].reshape(1,-1),model.wv[w1].reshape(1,-1))[0]
                    topic_score.append(word_score[0])
        
        topic_score = sum(topic_score)/len(topic_score) 
        model_score.append(topic_score)

    model_coherence = sum(model_score)/len(model_score) 
    print("k = ",k, ". Model coherence:", model_coherence)

    if model_coherence > max_model_coherence:
        max_model_coherence = model_coherence
        res_k = k

print("Best k:", res_k)
```

The analysis showed that `k = 11` yielded the highest model coherence. This is the number of topics to move forward with the final NMF model.

```
k =  10 . Model coherence: 0.0023823067576934894
k =  11 . Model coherence: 0.005569110116498036
k =  12 . Model coherence: 0.0019128068626202918
...
Best k: 11
```

### Non-Negative Matrix Factorization (NMF)

With the optimal number of topics (`res_k = 11`) determined, we applied NMF. NMF factorizes our document-term matrix **V** into two non-negative matrices, **W** and **H**: **V = W * H**

*   **H** (components matrix): Represents the topics, where each row (k) is a topic defined by non-negative weights for each term, the columns represent the terms. Sorting these weights gives us the most important terms for each topic.
*   **W** (document-topic matrix): With k columns, represents the membership weights of each document (line) in each topic.
 
![NMF](https://github.com/user-attachments/assets/d69e328a-60d2-4e80-9080-eb5eeb61b3b2){: .mx-auto.d-block :}

**Fig. 1.** Schematic diagram of NMF.

```python
nmf_model = NMF(res_k, random_state=1) 
w = nmf_model.fit_transform(a)
```

### Results: Unveiling the Topics!

In the top 10 terms for each of the 11 topics there are underlying themes. Here's what was found:

*   **Topic 0 -** technology, phone, video, speed, generation, device, network, broadband, image, picture - **Technology & Digital Agenda**
*   **Topic 1 -** club, player, football, team, chelsea, game, season, manager, champion, league - **Football/Sports**
*   **Topic 2 -** election, blair, party, minister, government, leader, tory, secretary, chancellor, democrat - **Politics & Elections**
*   **Topic 3 -** music, band, song, rock, artist, album, singer, record, single, award - **Music & Entertainment**
*   **Topic 4 -** forsyth, frederick, terrorist, internment, forsythe, totalitarianism, qaeda, fundamentalism, churchill, liberty - **Conflict/Terrorism (Specific Events)**
*   **Topic 5 -** growth, economy, market, price, rate, rise, bank, investment, analyst, dollar - **Economy & Finance**
*   **Topic 6 -** angel, rhapsody, bland, brit, guy, pulp, cheesy, deserve, joss, joke - **Entertainment/Pop Culture (Specific References)**
*   **Topic 7 -** sub, minute, goal, ball, yard, header, kick, cech, duff, cross - **Football/Match Details**
*   **Topic 8 -** software, virus, user, mail, program, computer, security, site, information, attack - **Cybersecurity & Software**
*   **Topic 9 -** court, yukos, bankruptcy, gazprom, case, fraud, russia, rosneft, khodorkovsky, unit - **Business/Legal (Specific Cases)**
*   **Topic 10 -** film, actor, award, oscar, star, actress, comedy, movie, nomination, ceremony - **Film & Awards**

These topics are remarkably coherent. For example, Topic 1 is clearly about technology, Topic 5 about economics, and Topic 10 about movies. There are even two distinct football-related topics (general football vs. match specifics) and two entertainment topics.

To further validate the model, the documents with the highest weights for each topic were examined. Since the filenames in the sample data included topic labels, it was possible to directly verify whether the model's assigned topics aligned with the actual content. The results showed a clear alignment. Documents categorized as `tech_xxx.txt` consistently appeared in the technology topic, `football_xxx.txt` in the football topics, `businessxxx.txt` in the business/economy topics, and `entertainment_xxx.txt` in the music/film topics. This strong correspondence verifies the validity of the NMF model.

### Comparative Analysis: NMF vs. LDA

The original paper highlighted NMF's ability to uncover "niche topics" and produce more semantically coherent results compared to traditional LDA. A basic LDA implementation was performed to observe this difference firsthand.

#### LDA Implementation

```python
# Pre-processing the speeches for LDA
text_tokens = []
for filename in files:
    with open(filename, 'rb') as f:
        lines = f.readlines()
        sup_list = []
        for line in lines:
            for token in re.split('\W+', str(line)):
                token = token.lower()
                if len(token) > 3 and not token.isnumeric() and not token.lower() in stopwords.words('english'):
                    sup_list.append(token)
    text_tokens.append(sup_list)

for doc in text_tokens:
    doc = [wordnet_lemmatizer.lemmatize(x) for x in doc]

# Turn the tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(text_tokens)

# Convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in text_tokens]

# We will use the same number of topics as in the NMF so we can compare
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=11, id2word=dictionary, passes=20)

# Printing the results
for idx, topic in ldamodel.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic} \n")
```

#### LDA Topics:

```
Topic: 0 
Words: 0.014*"said" + 0.012*"people" + 0.006*"could" + 0.005*"games" + 0.005*"technology" + 0.005*"users" + 0.004*"microsoft" + 0.004*"also" + 0.004*"online" + 0.004*"software" 

Topic: 1 
Words: 0.022*"said" + 0.013*"would" + 0.011*"government" + 0.009*"labour" + 0.008*"election" + 0.008*"blair" + 0.008*"party" + 0.006*"people" + 0.005*"also" + 0.005*"howard" 

... (and so on for all 11 topics)
```

As you can see, the LDA topics frequently include generic terms like "said," "would," and "also." These are common connectors and verbs that don't contribute much to defining a specific topic. In contrast, NMF was able to move past this linguistic "noise" and identify much more fine-grained and semantically coherent topics. This observation strongly supports the original paper's claim about NMF's advantage in capturing niche topics and producing more interpretable results.

### NMF's Power in Unstructured Text

This replication exercise clearly demonstrated the effectiveness of Non-Negative Matrix Factorization for topic modeling in unstructured text data. By meticulously pre-processing the text, carefully selecting the optimal number of topics using coherence measures, and applying NMF, distinct and semantically meaningful themes were successfully extracted.

The comparison with LDA further underscored NMF's strength, particularly in its ability to filter out generic terms and truly capture the essence of different topics. This makes NMF a powerful tool for researchers and analysts looking to understand large corpora of documents, whether they are political speeches, news articles, or any other form of text.

### Analysing the results of the paper

Between 1999 and 2014, over 210,000 speeches delivered in English by 1,735 Members of the European Parliament (MEPs) across 28 nations were analyzed to uncover the evolving political agenda of the European Parliament.

![NMF](https://github.com/user-attachments/assets/41b85b7b-5951-42b2-a21b-b7c79db66f91){: .mx-auto.d-block :}

**Fig. 2.** Evolution of terms describing the topic.

The evolution of key terms within topics shows how certain issues gained or lost prominence. Interestingly, even as terminology changed, shared core terms persisted—demonstrating the semantic validity of the results and highlighting stable thematic undercurrents within parliamentary debates.

![NMF](https://github.com/user-attachments/assets/e53a92df-da20-4a76-9aca-d01cb3717f8e){: .mx-auto.d-block :}

**Fig. 3.** Response to Exogenous Shocks.

External economic shocks significantly influenced parliamentary discussions. The timeline below marks major events that shaped the discourse:
* A - Start of the global financial crisis
* B - Revelations of Greece’s hidden debt
* C - Ireland receives liquidity injections
* D - ECB declares it will do "whatever it takes" to save the euro

These moments led to sharp changes in topic emphasis, reflecting urgent political and economic concerns.

![NMF](https://github.com/user-attachments/assets/fe39c5b3-f59a-45d9-ac0d-c60be1e01440){: .mx-auto.d-block :}

**Fig. 4.** Treaty Revisions in Discourse.

Treaty revisions were infrequent but significant in parliamentary rhetoric, appearing only sporadically throughout the 15-year period:
* A - Ireland’s rejection of the Treaty of Nice
* B - Negotiation of the Constitutional Treaty
* C - EU Enlargement Treaty
* D - Signing of the Lisbon Treaty

These key institutional changes triggered brief but notable shifts in political focus.

Dynamic topic modeling provides a powerful lens to trace the semantic and political evolution of the European Parliament. Through this method, researchers can identify how major events, both expected and unexpected, have shaped the political narrative across Europe’s most diverse legislative body.



