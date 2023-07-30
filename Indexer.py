import glob
import matplotlib.pyplot as plt 
import numpy as np
from collections import OrderedDict

class Indexer:
    def __init__(self, file_path="dataset/", format=".dat"):
        self.file_list=glob.glob(file_path+"*"+format)


    def get_dict(self):
        '''
        Return a dictionary where the keys are the terms and the values 
        are sorted lists of DocIDs.
        The DocID are generated sequentially during the parsing.
        '''
        dict = {}
        docid = 0         # counter to assign docids
        tot_docs = 0      # total number of docs
        tot_tokens = 0    # total number of tokens
        tot_postings = 0  # total number of postings
        
        for f in self.file_list:
            count_doc = 0       # number of docs per file .dat     
            count_tokens = 0    # number of tokens per file .dat
            count_postings = 0  # number of postings per file .dat

            with open(f, 'r') as file:              
                # Build dict{} of (key:value) pairs
                # key <-  term
                # value <- a list:  [cf, [doc_i1, doc_i2, doc_i3, ...]]
                #          where cf = collection frequency of the term
                #                [doc_i1, doc_i2, ...] are an integer list of DocIDs, 
                #                                      sorted by construction

                for line in file:
                    ll = line.split()

                    if len(ll) != 0:
                        if ll[0] == ".I":         # .I 1
                            count_doc = count_doc + 1
                            docid += 1  # assign sequentially the DocIDs
                            # docid = int(ll[1])  #  docid in the collection
                        elif ll[0] != ".W":       #  != .W  :   <textline> 
                            count_tokens += len(ll)
                            for el in ll:
                                if (el in dict):
                                    lenl = len(dict[el][1])
                                    dict[el][0] += 1
                                    if docid != dict[el][1][lenl-1]:  #doc_id unique
                                        dict[el][1].append(docid)
                                        count_postings += 1
                                else:
                                    dict[el] = [1, [docid]]  
                                            # value for key=el:
                                            #     [coll_frequency, [docid1, docid2, ....]]
                                    count_postings += 1

            print(f, " - docs: ", count_doc, "   tokes: ", count_tokens)
            tot_docs += count_doc
            tot_tokens += count_tokens
            tot_postings += count_postings
            
        print("Total no. of terms (Voc. SZ):", len(dict))
        print("Total no. of tokens:", count_tokens)
        print("Total no. of documents:", tot_docs)
        print("Total no. of postings:", tot_postings)
        
        return dict

    
    @staticmethod
    def create_standard_index(dict: dict, path="output/", file_lexicon="lexicon.idx", file_postings="postings.idx"):
        ''' 
        dict: dictionary containing for each term (key) 
            a list of docIDs (postings list of integers)
        file_lexicon: text file containing for each line a term, the collection frequency, and the length of the postings list 
                    associated with the term
        file_postings: text file that contains, one for line, the postings list of each term 
                    (a sorted list of integer docIDs).    
        '''
        with open(path+file_lexicon, "w") as f1, open(path+file_postings, "w") as f2:
            index = 0
            voc = list(dict.keys())
            voc.sort()
            for term in voc:
                f1.write("{} {} {}\n".format(term, dict[term][0], index))
                for docid in dict[term][1]:
                    f2.write("{} ".format(docid))
                f2.write("\n")
                index = index + len(dict[term][1])
        

    @staticmethod
    def create_clustering_based_index(dict: dict, path="output/", file_lexicon="lexicon.idx", file_postings="postings.idx"):
        pass

    

    @staticmethod
    def read_index(path="output/", file_lexicon="lexicon.idx", file_postings="postings.idx"):
        '''
        file_lexicon:  text file containing for each line a term, the collection frequency, and the length of the postings list 
                    associated
        file_postings: text file that contains, one for line, the postings list of each term 
                    (a sorted list of integer docIDs).    

        RETURN: a dictionary containing for each term (key) a list of docIDs (postings list of integers)
        '''
        new_dict = OrderedDict()
        with open(path+file_lexicon, 'r') as f1, open(path+file_postings, 'r') as f2:
            for line in f1:
                i += 1
                p = f2.readline()
                l = line.split() # two elements: (1) term, (2) collection frequency, and (2) associated postings list length
                pl = p.split()
                new_dict[l[0]] = [l[1], [int(el) for el in pl]]  # list of integers
        return(new_dict)
        

    @staticmethod
    def get_total_VB_enc_size(index: dict):
        pass


    @staticmethod
    def plot_Zip_law(dict: dict):

        freqs = []
        for v in dict.values():
            freqs.append(int(v[0]))

        freqs.sort(reverse=True)

        x = [i for i in range(1,len(freqs)+1)]

        x1 = np.log10(x)
        y1 = np.log10(freqs)

        # Zip law, fixed "slope" of the straight line in log.log = -1
        slope = -1
        intercept = np.mean(y1 + x1)   # value of log(K) in log.log

        # 10^(log10(cf)) = 10^(-log10(i) + intercept)
        # cf = 10^(-log10(i)) * 10^(intercept)
        # cf = -i * 10^(intercept)
        fitted_f = [(elx**slope) * (10**intercept) for elx in x]
        
        # plot the long tail, without log/log transformation
        plt.figure(1)
        plt.xlabel('terms (the 1000 most frequent ones)')
        plt.ylabel('freqs')
        lab = 'K: %.2f' % (10**intercept) + '      [' + '%.2f' % (10**intercept) + ' * i^(-1)]'
        plt.plot(x[:1000], freqs[:1000], 'r')
        plt.plot(x[:1000], fitted_f[:1000], 'b', label=lab)
        plt.legend()
        plt.show()

        # plot the same figure in log.log
        plt.figure(2)
        plt.xlabel('log_10(terms)')
        plt.ylabel('log_10(freqs)')
        lab = 'K: %.2f' % (10**intercept) + '      [' + 'log %.2f' % (10**intercept) + ' - log i]'
        plt.loglog(x, freqs, 'r')
        plt.loglog(x, fitted_f, 'b', label=lab)
        plt.legend()
        plt.show()




