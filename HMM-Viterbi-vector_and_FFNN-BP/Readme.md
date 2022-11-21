## ________________________________DIRECTIONS________________________________
1. Run all the cells.
2. Input for testing can be provided in cell 13.
3. The code treats upper and lower case indifferently (converts everything to lower case)
4. Below is the description of every part of the code.
5. Pre requisites: The libraries in '# Importing libraries' should be installed and running on system.


## ________________________________DESCRIPTION________________________________

## Part 1

Given a sequence of words, produce the POS tag sequence  
Technique to be used: HMM-Viterbi-vector (vector based; the whole corpus is corpus of word vectors which replace words)  
Use Universal Tag Set (12 in number);  
<ADJ, ADP, ., NOUN, CONJ, NUM, PRT, ADV, X, VERB, PRON, DET>  
5-fold cross validation  
Compare with HMM-Viterbi-symbolic   

## Part 2

Given a sequence of words, produce the POS tag sequence  
Technique to be used: word2vec vectors, FFNN and BP (a slide on FFNN-BP architecture is a must)  
Use Universal Tag Set (12 in number);  
<ADJ, ADP, ., NOUN, CONJ, NUM, PRT, ADV, X, VERB, PRON, DET>  
5-fold cross validation  
Compare with HMM-Viterbi-symbolic  

### FFNN-BP Architecture
![image](https://user-images.githubusercontent.com/89626355/203066018-6f1b85de-7a1b-4f58-bc96-cc99f9e5144f.png)

### Confusion Matrices (1 of 3) – HMM-Viterbi-symbolic
![image](https://user-images.githubusercontent.com/89626355/203066187-3ed5f989-c2ab-4069-b2ce-4e657deda353.png)

### Confusion Matrices (2 of 3) – HMM-Viterbi-vector
![image](https://user-images.githubusercontent.com/89626355/203066263-f66fdf40-a9a4-4fab-be40-f36b3f765565.png)

#### Confusion Matrices (3 of 3) – word2vec, FFNN, BP
![image](https://user-images.githubusercontent.com/89626355/203066325-0c884094-26d1-49db-87cd-b8ad88a56c6b.png)

#### Per POS Performance
![image](https://user-images.githubusercontent.com/89626355/203066448-431988cb-daf5-42b0-b363-254a71205349.png)

#### Overall Performance
![image](https://user-images.githubusercontent.com/89626355/203066528-02d72d8f-50b3-4759-8fc3-43423f92fe1b.png)

### Interpretation of Confusion (Error Analysis)
![image](https://user-images.githubusercontent.com/89626355/203066689-d6179257-3f0b-44e5-aa71-af2cda41a3e2.png)

![image](https://user-images.githubusercontent.com/89626355/203067348-7d2c768f-a37c-4f72-a91e-bf94338e952d.png)
![image](https://user-images.githubusercontent.com/89626355/203067387-3b9c97ab-cb92-4e6c-96b9-c511e540b8f4.png)
![image](https://user-images.githubusercontent.com/89626355/203067411-db39680b-6cd9-4c7a-a02e-26adcc85ed1d.png)

### Data Processing and Data Sparsity

To obtain the word vectors we used the Glove embedding lookup file given on https://github.com/stanfordnlp/GloVe  
For solving the problem of unseen words we use cosine similarity of vector In 1st method HMM-Viterbi Vector
