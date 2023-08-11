# clustering_for_inverted_index
File downloaded from: www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/

- pip install psutil
- pip install ortools

## Data collection RCV1

To use directly the *RCV1* collection is needed to buy CD-ROMs from Reuters, Ltd. and sign a research agreement.
However, Reuters have stated that distributing term/document matrices is not a violation of the Agreement.

To ensure that the original data cannot be reconstructed, the term/document matrices distributed are built by removing words from a large stop list (including essentially all linguistic function words), replacing the remaining words with stems, and scrambling the order of the stems.

Specifically, Appendix 12 - B.12.i. RCV1-v2 Token Files  consists of **five** ASCII files containing tokenized documents. 

The files fall in two groups, and the names and the group purposes depend on the usage of RCV1 for text classification, a specific machine learning (ML) task. So the files are subdivided in training and test, as usual for ML predictive tasks.


### File format

Each document in a file is represented in a format used by the SMART text retrieval system. A document has the format:

      .I <docid>
      .W
      <textline>+
      <blankline>


where:

      <docid> : Reuters-assigned document id. 
      <textline> : A line of white-space separated strings, one for each token 
                   produced by preprocessing for the specified document. 
                   These lines never begin with a period followed by an upper case alphabetic character.
      <blankline> : A single end of line character. 

This is an example of a list of (short) tokenized documents in a given file:

      .I 1
      .W
      now is the time for all good documents
      to come to the aid of the ir community
      
      .I 2
      .W
      i am the best document since i have only one line

      .I 3
      .W
      no i am the best document 


For more detail, see the distribution page: http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm


[LYRL2004] *Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf. *
