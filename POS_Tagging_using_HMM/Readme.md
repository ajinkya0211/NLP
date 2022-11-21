## ________________________________DIRECTIONS________________________________
1. Run all the cells.
2. Input for testing can be provided in cell 13.
3. The code treats upper and lower case indifferently (converts everything to lower case)
4. Below is the description of every part of the code.
5. Pre requisites: The libraries in '# Importing libraries' should be installed and running on system.


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
