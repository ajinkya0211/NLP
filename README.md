# POS_tagging

Here I have implemented and compared three methods of POS tagging, namely:  
1. Hidden Markov Model (HMM)
2. Hidden Markov Model (HMM) using Viterbi vector and Feed Forward Neural Network (FFNN) with Backpropogation (BP)

Below are the overall results:
![image](https://user-images.githubusercontent.com/89626355/203076680-1c359e90-33b1-488d-9b57-e300d91ebc7b.png)

A detailed analysis of models can be found in Readme files in given folders


## ________________________________DIRECTIONS________________________________
1. Run all the cells.
2. Input for testing can be provided in cell 13.
3. The code treats upper and lower case indifferently (converts everything to lower case)
4. Below is the description of every part of the code.
5. Pre requisites: The libraries in '# Importing libraries' should be installed and running on system.

# 1. Hidden Markov Model (HMM)

## ________________________________DESCRIPTION________________________________


### Problem statement

Given a sequence of words, produce the POS tag sequence  

Technique to be used: HMM-Viterbi  

Use Universal Tag Set (12 in number)  

5-fold cross validation   

<ADJ, ADP, ., NOUN, CONJ, NUM, PRT, ADV, X, VERB, PRON, DET>  


### Importing libraries

The libraries used are:  
nltk - for dataset and tagset  
numpy, pandas, tqdm and scikit-learn for dealing with data  
seaborn, matplotlib - for visualization and plotting  


### DataSet

For accessing data I have used nltk library: brown.tagged_sents(tagset='universal') and split used is 80% training & 20% testing.  

The dataset is splitted and functions are created for:  
1. Determining length of a list from data  
2. Inserting tags  
3. Retreiving tags  
  
A variable, alpha for handling some cases is also defined  

Only lower case is used and I store data by creating a class where I store tokens in dict(), in index2value and value2index for each word as well as for tags.  
For creating 3 matrix I used the numpy library.  
Transition = count(tagi-1,tagi)/count(tagi) (# two tag appear together divided by total count of that tag)  
Emission = count(tag,word)/count(tag)(# word and tag appear together divided by total count of that tag)  
To handle unknown word I did Laplace Smoothing  



### HMM Model

I have defined functions to get words (get_word) and tags (get_tag) from storage and universal tagset  

Then I proceed to extract transmission matrix, emission matrix and probability of tags from the data implementing HMM  


### Vertibi algorithm

Vertibi algorithm is implememnted to get Part Of Speech tag for words in input.  
It includes vertibi initialization, iteration and sequence identification as told in class  
These are explained as:  
Data Structure use: a) SeqScore N*T Dimension array(N= #tags,T=length of o/p sequence), b)Another N*T array called backpointer to recover path  
Step of viterbi Implementation:  
1) Initialization: In this I have initialised both the matrix with zeros  
2) Iteration: for p in range(1,sent_length)  
                  for i in range(tag_id)  
                      SeqScore[i,p] = transition_matrix[ np.argmax(np.multiply(SeqScore[:, p-1], transition_matrix[:, i]))]* emission_matrix[i,word_id]  
                      Back_pointer[i,p]=np.argmax(np.multiply(SeqScore[:, p-1], transition_matrix[:, i]))  
3) Sequence Identification:  
c(T) = i that maximize SeqScore(i,T)  
For i from (T-1) to 1 do  
c(i) = Back_pointer[c(i+1),(i+1)]   

refence: https://www.youtube.com/watch?v=AwQ5nUB119s  


### Test

Here input can be given to predict tags from the model.  
Extra delimiter has been introduced before punctuation marks to tackle the error of counting the whole word attached to punctuation as punctuation by the model.  


### Cross Validation

Confusion matrix (Interpretation and error analysis):  
I then derive the confusion matrix using prediction function and it has been plotted as a heatmap for better visualization.  

#### Confusion Matrix (12 X 12)
![image](https://user-images.githubusercontent.com/89626355/203062891-deedc6ae-3267-4fb0-bf33-060230a98034.png)

#### Interpretation of confusion (error analysis)

Given that the actual tag is X, the probability that it has been misclassified as a NOUN is maximum, 0.306.  
Also, from the confusion matrix of X and NOUN, the error in classification is obtained as 0.0021.  
Reason:   
X : foreign words, typos, abbreviations  
Number of nouns in the dataset is maximum  
Therefore, the probability of a word being tagged as a noun is the highest by the algorithm  
Hence, most of the words which are to be tagged X in actual is being tagged as a NOUN  
  
5-fold-cross validation is used to quantitatively measure the working of the function  

#### Per POS Performance

![image](https://user-images.githubusercontent.com/89626355/203062382-037c16c0-d515-446f-b8df-f203aba44a0c.png)

#### Overall Performance

The observed overall parameters are:  
Precision = 96.02  
Recall = 96.05  
F-score (3 values)  
F1-score = 96.03  
F0.5-score = 96.03  
F2-score = 96.04  
These are calculated by taking weighted averages of data from individual tags.  
  
Overall accuracy of the model is 96.02

# 2. Hidden Markov Model (HMM) using Viterbi vector and Feed Forward Neural Network (FFNN) with Backpropogation (BP)

## ________________________________DIRECTIONS________________________________
1. Run all the cells.
2. Input for testing can be provided in cell 13.
3. The code treats upper and lower case indifferently (converts everything to lower case)
4. Below is the description of every part of the code.
5. Pre requisites: The libraries in '# Importing libraries' should be installed and running on system.


## ________________________________DESCRIPTION________________________________

## Library used are:
1) Keras
2) sklearn
3) pandas
4) Numpy 
5) MatPlotLib
6) seaborn
7) tqdm
8) gensim

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
