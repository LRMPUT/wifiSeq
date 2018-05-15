/*
	Copyright (c) 2015,	TAPAS Team:
	-Jan Wietrzykowski (jan.wietrzykowski@cie.put.poznan.pl).
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

#ifndef UNIONFIND_H_
#define UNIONFIND_H_

#include <vector>
#include <cstddef>

/** \brief Struktura reprezentująca węzeł w klasie UnionFind.
 *
 */
struct SetNode{
	int parent, rank, nsize;
	SetNode() : parent(-1), rank(0), nsize(1) {}
	SetNode(int iparent, int irank, int insize) : parent(iparent), rank(irank), nsize(insize) {}
};

/** \brief Klasa reprezentująca rozłączne zbiory, umożliwiająca
 * 			efektywne ich łączenie.
 */
class UnionFind{
	std::vector<SetNode> set;
public:
	UnionFind(int icount);
	~UnionFind();

	/** \brief Funkcja znajdująca id zbioru, do którego należy węzeł node.
	 *
	 */
	int findSet(int node);

	/** \brief Funkcja łącząca dwa zbiory.
	 *
	 */
	int unionSets(int node1, int node2);

	/** \brief Funkcja zwracająca rozmiar zbioru.
	 *
	 */
	int size(int node);
};


#endif /* UNIONFIND_H_ */
