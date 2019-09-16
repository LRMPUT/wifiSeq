/*
	Copyright (c) 2019,	Mobile Robots Laboratory:
	-Jan Wietrzykowski (jan.wietrzykowski@put.poznan.pl).
	Poznan University of Technology
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification,
	are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
	THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
	AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "pgm/UnionFind.h"

UnionFind::UnionFind(int icount){
	set.resize(icount);
}

UnionFind::~UnionFind(){

}

int UnionFind::findSet(int node){
	if(set[node].parent == -1){
		return node;
	}

	set[node].parent = findSet(set[node].parent);
	return set[node].parent;
}

int UnionFind::unionSets(int node1, int node2){
	int node1Root = findSet(node1);
	int node2Root = findSet(node2);
	if(set[node1Root].rank > set[node2Root].rank){
		set[node2Root].parent = node1Root;
		set[node1Root].nsize += set[node2Root].nsize;
		return node1Root;
	}
	else if(set[node1Root].rank < set[node2Root].rank){
		set[node1Root].parent = node2Root;
		set[node2Root].nsize += set[node1Root].nsize;
		return node2Root;
	}
	else if(node1Root != node2Root){
		set[node2Root].parent = node1Root;
		set[node1Root].rank++;
		set[node1Root].nsize += set[node2Root].nsize;
		return node1Root;
	}
	return -1;
}

int UnionFind::size(int node){
	return set[findSet(node)].nsize;
}
