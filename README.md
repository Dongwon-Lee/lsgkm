## LS-GKM: A new gkm-SVM software for large-scale datasets

gkm-SVM, a sequence-based method for predicting regulatory DNA elements,
is a useful tool for studying gene regulatory mechanisms.
In continuous efforts to improve the method, new software, `LS-GKM`,
is introduced.  It offers much better scalability and provides further
advanced gapped *k*-mer based kernel functions.  As a result, LS-GKM
achieves considerably higher accuracy than the original gkm-SVM.

### Citation

*Please cite the following paper if you use LS-GKM in your research:*

* Ghandi, M.†, Lee, D.†, Mohammad-Noori, M. & Beer, M. A. Enhanced Regulatory Sequence Prediction Using Gapped k-mer Features. PLoS Comput Biol 10, e1003711 (2014). doi:10.1371/journal.pcbi.1003711 *† Co-first authors*

* Lee, D. LS-GKM: A new gkm-SVM for large-scale Datasets. Bioinformatics btw142 (2016). doi:10.1093/bioinformatics/btw142


### Installation

After downloading and extracting the source codes, type:

    $ cd src
    $ make 

If successful, You should be able to find the following executables in the current (src) directory:

    gkmtrain
    gkmpredict

`make install` will simply copy these two executables to the `../bin` direcory


### Tutorial

We introduce the users to the basic workflow of `LS-GKM`.  Please refer to help messages 
for more detailed information of each program.  You can access to it by running the programs 
without any argument/parameter.
  

#### Training of LS-GKM

You train a SVM classifier using `gkmtrain`. It takes three arguments; 
positive sequence file, negative sequence file, and prefix of output.


    Usage: gkmtrain [options] <posfile> <negfile> <outprefix>

     train gkm-SVM using libSVM

    Arguments:
     posfile: positive sequence file (FASTA format)
     negfile: negative sequence file (FASTA format)
     outprefix: prefix of output file(s) <outprefix>.model.txt or
                <outprefix>.cvpred.txt

    Options:
     -t <0 ~ 5>   set kernel function (default: 4 wgkm)
                  NOTE: RBF kernels (3 and 5) work best with -c 10 -g 2
                    0 -- gapped-kmer
                    1 -- estimated l-mer with full filter
                    2 -- estimated l-mer with truncated filter (gkm)
                    3 -- gkm + RBF (gkmrbf)
                    4 -- gkm + center weighted (wgkm)
                         [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]
                    5 -- gkm + center weighted + RBF (wgkmrbf)
     -l <int>     set word length, 3<=l<=12 (default: 11)
     -k <int>     set number of informative column, k<=l (default: 7)
     -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)
     -g <float>   set gamma for RBF kernel. -t 3 or 5 only (default: 1.0)
     -M <int>     set the initial value (M) of the exponential decay function
                  for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)
     -H <float>   set the half-life parameter (H) that is the distance (D) required
                  to fall to half of its initial value in the exponential decay
                  function for wgkm-kernels. -t 4 or 5 only (default: 50)
     -R           if set, reverse-complement is not considered as the same feature
     -c <float>   set the regularization parameter SVM-C (default: 1.0)
     -e <float>   set the precision parameter epsilon (default: 0.001)
     -w <float>   set the parameter SVM-C to w*C for the positive set (default: 1.0)
     -m <float>   set cache memory size in MB (default: 100.0)
                  NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended
     -s           if set, use the shrinking heuristics
     -x <int>     set N-fold cross validation mode (default: no cross validation)
     -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)
     -r <int>     set random seed for shuffling in cross validation mode (default: 1)
     -v <0 ~ 4>   set the level of verbosity (default: 2)
                    0 -- error msgs only (ERROR)
                    1 -- warning msgs (WARN)
                    2 -- progress msgs at coarse-grained level (INFO)
                    3 -- progress msgs at fine-grained level (DEBUG)
                    4 -- progress msgs at finer-grained level (TRACE)
    -T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16
                     (default: 1)


First try to train a model using simple test files. Type the following command in `tests/` directory:

    $ ../bin/gkmtrain wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.tr.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.tr.fa test_gkmtrain

It will generate `test_gkmtrain.model.txt`, which will then be used for scoring of 
any DNA sequences as described below.  This result should be the same as `wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt`

You can also perform cross-validation (CV) analysis with `-x <N>` option. For example,
the following command will perform 5-fold CV. 

    $ ../bin/gkmtrain -x 5 wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.tr.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.tr.fa test_gkmtrain

The result will be stored in `test_gkmtrain.cvpred.txt`, and this should be the same as 
`wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.cvpred.txt`

Please note that it will run SVM training *N* times, which can take time if training 
sets are large.  In this case, you can perform CV analysis on a specific set 
by using `-i <I>` option for parallel runnings. The output will be `<outprefix>.cvpred.<I>.txt`

The format of the cvpred file is as follows:
  
    [sequenceid] [SVM score] [label] [CV-set]
    ...


#### Scoring DNA sequence using gkm-SVM

You use `gkmpredict` to score any set of sequences.

    Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>

     score test sequences using trained gkm-SVM

    Arguments:
     test_seqfile: sequence file for test (fasta format)
     model_file: output of gkmtrain
     output_file: name of output file

    Options:
     -v <0|1|2|3|4>  set the level of verbosity (default: 2)
                       0 -- error msgs only (ERROR)
                       1 -- warning msgs (WARN)
                       2 -- progress msgs at coarse-grained level (INFO)
                       3 -- progress msgs at fine-grained level (DEBUG)
                       4 -- progress msgs at finer-grained level (TRACE)
    -T <1|4|16>      set the number of threads for parallel calculation, 1, 4, or 16
                     (default: 1)

Here, you will try to score the positive and the negative test sequences. Type:

    $ ../bin/gkmpredict wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.test.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt test_gkmpredict.txt
    $ ../bin/gkmpredict wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.test.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt test_gkmpredict.neg.txt


#### Generating weight files for deltaSVM

You need to generate all possible non-redundant *k*-mers using the Python script
`scripts/nrkmers.py`.  Then, you score them using `gkmpredict` as described above. 
The output of `lgkmpredict` can be directly used by the deltaSVM script `deltasvm.pl`
available from our deltasvm website.

** Please email Dongwon Lee (dongwon.lee AT childrens DOT harvard DOT edu) if you have any questions. **
