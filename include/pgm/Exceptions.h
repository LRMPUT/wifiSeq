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

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <exception>
#include <string>
#include <sstream>
#include <typeinfo>
#ifdef __linux__
#include <execinfo.h>
#endif

#define PGM_EXCEPTION(msg) pgm_exception(__FILE__, __LINE__, string(msg))

class pgm_exception : public std::exception
{
public:
	pgm_exception(const std::string& ifile,
				const size_t iline,
				const std::string& imsg = std::string(),
				const std::string& exName = "pgm_exception")
		: msg(imsg),
		  file(ifile),
		  line(iline)
	{
		createWhatMsg(exName);
	}

	pgm_exception(const std::string& ifile,
				const size_t iline,
				const std::exception& srcEx,
				const std::string& exName = "pgm_exception")
		: file(ifile),
		  line(iline)
	{
		msg = "wrapped exception class '" + std::string(typeid(srcEx).name()) + "': " + std::string(srcEx.what());
		createWhatMsg(exName);
	}
	virtual ~pgm_exception() throw() {};

	const char* what() const throw() {
		return whatMsg.c_str();
	}

	const char* get_file() const throw() {
		return file.c_str();
	}

	size_t get_line() const throw() {
		return line;
	}

	virtual const char* get_message() {
		return msg.c_str();
	}

private:
	std::string msg;
	std::string file;
	size_t line;

	std::string whatMsg;

	void createWhatMsg(const std::string& ex_name){
		std::stringstream ss;
		ss << "Exception '" << ex_name << "' thrown in file '" << file << "' line '" << line << "' with message:\n" << get_message();

#ifdef __linux__
		void *buffer[100];
		char **strings;

		int nptrs = backtrace(buffer, 100);

		strings = backtrace_symbols(buffer, nptrs);
		if (strings != nullptr) {
			ss << std::endl << std::endl << "Stack trace:" << std::endl;
			for (int j = 0; j < nptrs; j++)
				ss << strings[j] << std::endl;
			free(strings);
		}
#endif
		whatMsg = ss.str();
	}
};


#endif /* EXCEPTIONS_H_ */
