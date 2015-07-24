// Additive Groves / ag_functions.h: declarations of Additive Groves global functions
// (c) Daria Sorokina

#include "definitions.h"
#include "TrainInfo.h"
#include "INDdata.h"

//saves a vector into a binary file
fstream& operator << (fstream& fbin, doublev& vec);

//saves a vector of vectors into a binary file
fstream& operator << (fstream& fbin, doublevv& mx);

//saves a vector of vectors of vectors into a binary file
fstream& operator << (fstream& fbin, doublevvv& trivec);

//reads a vector from a binary file
fstream& operator >> (fstream& fbin, doublev& vec);

//reads a vector of vectors from a binary file
fstream& operator >> (fstream& fbin, doublevv& mx);

//reads a vector of vectors of vectors from a binary file
fstream& operator >> (fstream& fbin, doublevvv& trivec);

//generates output files for train and expand commands
void trainOut(TrainInfo& ti, doublevv& dir, doublevvv& rmsV, doublevvv& surfaceV, doublevvv& predsumsV, 
			  int itemN, doublevv& dirStat, int startAlphaNo = 0, int startTiGNNo = 0);

//converts the number of a valid alpha value into the actual value
double alphaVal(int alphaNo);

//converts the number of a valid TiG value into the actual value
int tigVal(int tigNNo);

//rounds tigN down to the closest appropriate value
double adjustTiGN(int tigN);

//converts min alpha value into the number of alpha values
int getAlphaN(double minAlphaVal, int trainN);

//converts max tigN value into the number of tigN values
int getTiGNN(int tigN);

//converts number to string
std::string itoa(int value, int base);

//trains and saves a Layered Groves ensemble (Additive Groves trained in layered style)
double layeredGroves(INDdata& data, TrainInfo& ti, string modelFName);

//runs Layered Groves repeatN times, returns average performance and standard deviation, saves the last model 
double meanLG(INDdata& db, TrainInfo ti, int repeatN, double& resStd, string modelFName);

//implementation for erase for reverse iterator
void rerase(intv& vec, intv::reverse_iterator& iter);

