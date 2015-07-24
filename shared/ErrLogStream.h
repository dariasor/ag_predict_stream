// ErrLogStream.h: implementation of ErrLogStream class and << operator
// Redirects output to both cerr (console error output) and file log.txt
// (c) Daria Sorokina

#pragma once

#include <fstream>
#include <iostream>

class ErrLogStream
{
};	

template <class T>
ErrLogStream& operator << (ErrLogStream& errlogout, T& data)
{
	cerr << data;
	cerr.flush();

	return errlogout;
}
