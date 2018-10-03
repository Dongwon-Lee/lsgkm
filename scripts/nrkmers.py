#!/usr/local/bin/python
"""
	nrkmers.py: generate all possible non-redundant k-mers

	Copyright (C) 2014 Dongwon Lee

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys

def revcomp(seq):
	rc = {'A':'T', 'G':'C', 'C':'G', 'T':'A'}
	return ''.join([rc[seq[i]] for i in range(len(seq)-1, -1, -1)])

def id2kmer(kmerid, k):
	kmer = ''
	nts = ['A', 'C', 'G', 'T']
	for i in range(k):
		kmer = nts[(kmerid % 4)] + kmer
		kmerid = int(kmerid/4)

	return kmer

def main(argv = sys.argv):
	usage = "Usage: python nrkmers.py KMER_LENGTH OUTPUT"
	desc = "generate all possible non-redundant k-mers with KMER_LENGTH and save it in FASTA format."

	if len(sys.argv) != 3:
		print(usage)
		print("")
		print(desc) 
		print("")
		sys.exit(0)

	kmerlen = int(sys.argv[1])
	output = sys.argv[2]

	fout = open(output, 'w')
	kmers = set()
	for kid in range(4**kmerlen):
		kmer = id2kmer(kid, kmerlen)
		if kmer not in kmers:
			fout.write( ">" + kmer + "\n" + kmer + "\n" )
			kmers.add(revcomp(kmer))

if __name__=='__main__': main()
