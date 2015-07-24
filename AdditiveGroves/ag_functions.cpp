// Additive Groves / ag_functions.cpp: implementations of Additive Groves global functions
//
// (c) Daria Sorokina

#include "ag_functions.h"
#include "functions.h"
#include "LogStream.h"
#include "Grove.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

//generates all output for train, expand and merge commands except for saving the models themselves
void trainOut(TrainInfo& ti, doublevv& dir, doublevvv& rmsV, doublevvv& surfaceV, doublevvv& predsumsV, 
			  int itemN, doublevv& dirStat, int startAlphaNo, int startTiGNNo)
{
//Generate temp files that can be used by other commands later. (in addition to saved groves)

	int alphaN = getAlphaN(ti.minAlpha, itemN); //number of different alpha values
	int tigNN = getTiGNN(ti.maxTiGN);	//number of different tigN values
	LogStream clog;
	
	//params file is a text file
	fstream fparam;	
	fparam.open("./AGTemp/params.txt", ios_base::out); 
	//general set of parameters
	fparam << ti.seed << '\n' << ti.trainFName << '\n' << ti.validFName << '\n' << ti.attrFName 
		<< '\n' << ti.minAlpha << '\n' << ti.maxTiGN << '\n' << ti.bagN << '\n';
	//speed and directions matrix
	if(ti.mode == FAST)
	{
		fparam << "fast" << endl;

		if(!dir.empty())
		{
			fstream fdir;	
			fdir.open("./AGTemp/dir.txt", ios_base::out); 
			for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
			{
				for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
					fdir << dir[tigNNo][alphaNo] << '\t';
				fdir << endl;
			}
			fdir.close();
		}
	}
	else if(ti.mode == SLOW)
		fparam << "slow" << endl;
	else //if(ti.mode == LAYERED)
		fparam << "layered" << endl;

	//performance metric
	if(ti.rms)
		fparam << "rms" << endl;
	else
		fparam << "roc" << endl;

	fparam.close();

	//save rms performance matrices for every bagging iteration
	fstream fbagrms;	
	fbagrms.open("./AGTemp/bagrms.txt", ios_base::out); 
	for(int bagNo = 0; bagNo < ti.bagN; bagNo++)
	{
		fbagrms << endl;
		for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
		{
			for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
				fbagrms << rmsV[tigNNo][alphaNo][bagNo] << '\t';
			fbagrms << endl;
		}
	}
	fbagrms.close();

	//same for roc performance, if applicable
	if(!ti.rms)
	{
		fstream fbagroc;	
		fbagroc.open("./AGTemp/bagroc.txt", ios_base::out); 
		for(int bagNo = 0; bagNo < ti.bagN; bagNo++)
		{
			fbagroc << endl;
			for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
			{
				for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
					fbagroc << surfaceV[tigNNo][alphaNo][bagNo] << '\t';
				fbagroc << endl;
			}
		}
		fbagroc.close();
	}

	//save sums of predictions on the last bagging iteration
	fstream fsums;	
	fsums.open("./AGTemp/predsums.bin", ios_base::binary | ios_base::out);
	fsums << predsumsV;
	fsums.close();


//Generate the actual output of the program

	//performance table on the last iteration of bagging
	fstream fsurface;	//output text file with performance matrix
	fsurface.open("performance.txt", ios_base::out); 

	double bestPerf;					//best performance on validation set
	int bestTiGNNo, bestAlphaNo;		//ids of parameters that produce best performance
	int bestTiGN; double bestAlpha;		//parameters that produce best performance
	//output performance matrix, find best performance and corresponding parameter values
	for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
	{
		int tigN = tigVal(tigNNo);
		for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
		{
			double alpha;
			if(alphaNo < alphaN - 1)
				alpha = alphaVal(alphaNo);
			else //this is a special case because minAlpha can be zero
				alpha = ti.minAlpha;
			
			//output a single grid point with coordinates
			double& curPerf = surfaceV[tigNNo][alphaNo][ti.bagN - 1];
			fsurface << alpha << " \t" << tigN << " \t" << curPerf << endl;

			//check for the best result inside the active output area
			if((tigNNo >= startTiGNNo) && (alphaNo >= startAlphaNo))
				if((tigNNo == startTiGNNo) && (alphaNo == startAlphaNo) || 
					ti.rms && (curPerf < bestPerf) ||
					!ti.rms && (curPerf > bestPerf) ||
					((curPerf == bestPerf) && //if the result is the same, choose the less complex model
						(pow((double)2, alphaNo) * tigN < pow((double)2, bestAlphaNo) * bestTiGN))
					)	
				{
					bestTiGNNo = tigNNo;
					bestTiGN = tigN;
					bestAlphaNo = alphaNo;
					bestAlpha = alpha;
					bestPerf = curPerf;
				}
		}		
	}
	
	fsurface << "\n\n";

	//output the same number in the form of matrix
	for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
	{
		for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
			fsurface << surfaceV[tigNNo][alphaNo][ti.bagN - 1] << " \t";
		fsurface << endl;
	}
	fsurface.close();

	fstream fdirStat;	
	fdirStat.open("./AGTemp/dirstat.txt", ios_base::out);
	//output directions stats in the form of matrix
	for(int alphaNo = 0; alphaNo < alphaN; alphaNo++)
	{
		for(int tigNNo = 0; tigNNo < tigNN; tigNNo++)
			fdirStat << dirStat[tigNNo][alphaNo] / ti.bagN << " \t";
		fdirStat << endl;
	}
	fdirStat.close();


	//add bestPerf, bestTiGN, bestAlpha, bagN, trainN to AGTemp\best.txt 
		//bestPerf will be used in the output on the next iteration, others are used by ag_save.exe
	//best.txt file is a text file
	fstream fbest;
	fbest.open("./AGTemp/best.txt", ios_base::out); 
	fbest << bestPerf << '\n' << bestTiGN << '\n' << bestAlpha << '\n' << ti.bagN << '\n' << itemN 
		<< endl;
	fbest.close();	

	//output rms bagging curve in the best (alpha, TiGN) point
	fstream frmscurve;	//output text file 
	frmscurve.open("bagging_rms.txt", ios_base::out); 
	for(int bagNo = 0; bagNo < ti.bagN; bagNo++)
		frmscurve << rmsV[bestTiGNNo][bestAlphaNo][bagNo] << endl;
	frmscurve.close();

	//same for roc curve, if applicable
	if(!ti.rms)
	{
		fstream froccurve;	//output text file 
		froccurve.open("bagging_roc.txt", ios_base::out); 
		for(int bagNo = 0; bagNo < ti.bagN; bagNo++)
			froccurve << surfaceV[bestTiGNNo][bestAlphaNo][bagNo] << endl;
		froccurve.close();
	}

	//analyze whether more bagging should be recommended based on the rms curve in the best point
	bool recBagging = moreBag(rmsV[bestTiGNNo][bestAlphaNo]);
	//and based on the curve in a more complex point (if there is one)
	recBagging |= moreBag(rmsV[min(bestTiGNNo + 1, tigNN - 1)][min(bestAlphaNo + 1, alphaN - 1)]);
		
	//output results and recommendations
	clog << "Best model:\n\tAlpha = " << bestAlpha << "\n\tN = " << bestTiGN;
	if(ti.rms)
		clog << "\nRMSE on validation set = " << bestPerf << "\n";
	else
		clog << "\nROC on validation set = " << bestPerf << "\n";

	//if the best possible performance is not achieved,
	//and the best value is on the border, or bagging has not yet converged, recommend expanding
	if( (ti.rms && (bestPerf != 0) || !ti.rms && (bestPerf != 1)) &&
		(((bestAlpha == ti.minAlpha) && (ti.minAlpha != 0)) || (bestTiGN == ti.maxTiGN) || recBagging))
	{
		clog << "\nRecommendation: relaxing model parameters might produce a better model.\n"
			<< "Suggested action: ag_expand";
		if((bestAlpha == ti.minAlpha) && (ti.minAlpha != 0))
		{
			double recAlpha = ti.minAlpha * 0.1;
			//make sure that you don't recommend alpha that is too small for this data set
			if(1/(double)itemN >= recAlpha)
				recAlpha = 0;
			clog << " -a " << recAlpha;
		}
		if(bestTiGN == ti.maxTiGN)
		{
			int recTiGN = tigVal(getTiGNN(ti.maxTiGN) + 1);
			clog << " -n " << recTiGN;
		}
		if(recBagging)
		{
			int recBagN = ti.bagN + 40;
			clog << " -b " << recBagN;
		}
		clog << "\n";
	}
	else
		clog << "\nYou can save the best model for the further use. \n" 
			<< "Suggested action: ag_save -a " << bestAlpha << " -n " << bestTiGN << "\n";
	clog << "\n";
}

//saves a vector into a binary file
fstream& operator << (fstream& fbin, doublev& vec)
{
	int n = (int) vec.size();
	for(int i = 0; i < n; i++)
		fbin.write((char*) &(vec[i]), sizeof(double)); 
	return fbin;
}

//saves a matrix represented as vector of vectors into a binary file
fstream& operator << (fstream& fbin, doublevv& mx)
{
	int n = (int) mx.size();
	for(int i = 0; i < n; i++)
		fbin << mx[i];
	return fbin;
}

//reads a vector from a binary file
//vector should have the right size already
fstream& operator >> (fstream& fbin, doublev& vec)
{
	int n = (int) vec.size();
	for(int i = 0; i < n; i++)
		fbin.read((char*) &(vec[i]), sizeof(double)); 
	return fbin;
}

//reads a matrix represented as vector of vectors from a binary file
//all vectors should have the required size already
fstream& operator >> (fstream& fbin, doublevv& mx)
{
	int n = (int) mx.size();
	for(int i = 0; i < n; i++)
		fbin >> mx[i];
	return fbin;
}

//saves a vector of vectors of vectors into a binary file
fstream& operator << (fstream& fbin, doublevvv& trivec)
{
	int n = (int) trivec.size();
	for(int i = 0; i < n; i++)
		fbin << trivec[i];
	return fbin;

}

//reads a vector of vectors of vectors from a binary file
fstream& operator >> (fstream& fbin, doublevvv& trivec)
{
	int n = (int) trivec.size();
	for(int i = 0; i < n; i++)
		fbin >> trivec[i];
	return fbin;
}

//converts min alpha value into the number of alpha values
int getAlphaN(double minAlpha, int trainN)
{
	int alphaN; 
	for(alphaN = 0; 
		(minAlpha < alphaVal(alphaN) - 0.000000000000001) && //to adjust for rounding errors
			(1 / (double)trainN < alphaVal(alphaN) - 0.000000000000001); 
		alphaN++)
		;
	alphaN++;

	return alphaN;
}

//converts max tigN value into the number of tigN values
int getTiGNN(int tigN)
{
	int tigValue = 1;
	for(int tigNNo = 0; tigNNo < tigN + 1; tigNNo++)
		if(tigVal(tigNNo) > tigN)
			return tigNNo;

	return -1;
}

//Converts the number of a valid alpha value into the actual value
double alphaVal(int alphaNo)
{
	//ordered valid values: 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, ...  
	double alpha = pow(0.1, alphaNo / 3 + 1);
	if(alphaNo % 3 == 0)
		alpha *= 5;
	else if(alphaNo % 3 == 1)
		alpha *= 2;

	return alpha;
}

//Converts the number of a valid TiG value into the actual value
int tigVal(int tigNNo)
{
	return (int) (sqrt(pow(2.0, tigNNo + 1)) + 0.5);
}

//Rounds tigN down to the closest appropriate value
double adjustTiGN(int tigN)
{
	int tigNN = getTiGNN(tigN);
	return tigVal(tigNN - 1);
}

//The following code for linux-compatible string itoa is copied from the web site of Stuart Lowe
//http://www.jb.man.ac.uk/~slowe/cpp/itoa.html
/**
	
 * C++ version std::string style "itoa":
	
 */
	
std::string itoa(int value, int base) {
	

	
	enum { kMaxDigits = 35 };
	
	std::string buf;
	
	buf.reserve( kMaxDigits ); // Pre-allocate enough space.
	

	
	// check that the base if valid
	
	if (base < 2 || base > 16) return buf;
	

	
	int quotient = value;
	

	
	// Translating number to string with base:
	
	do {
	
		buf += "0123456789abcdef"[ std::abs( quotient % base ) ];
	
		quotient /= base;
	
	} while ( quotient );
	

	
	// Append the negative sign for base 10
	
	if ( value < 0 && base == 10) buf += '-';
	

	
	std::reverse( buf.begin(), buf.end() );
	
	return buf;
}


//trains a Layered Groves ensemble (Additive Groves trained in layered style) 
//if modelFName is not empty, saves the model
//returns performance on validation set
double layeredGroves(INDdata& data, TrainInfo& ti, string modelFName)
{
	doublev validTar; //true response values on validation set
	int validN = data.getTargets(validTar, VALID); 
	doublev predsumsV(validN, 0); 	//sums of predictions for each data point
	
	if(!modelFName.empty())
	{//save the model's header
		fstream fmodel(modelFName.c_str(), ios_base::binary | ios_base::out);
		fmodel.write((char*) &ti.mode, sizeof(enum AG_TRAIN_MODE));
		fmodel.write((char*) &ti.maxTiGN, sizeof(int));
		fmodel.write((char*) &ti.minAlpha, sizeof(double));
		fmodel.close();		
	}

	//build bagged models, calculate sums of predictions
	for(int bagNo = 0; bagNo < ti.bagN; bagNo++)
	{
		cout << "\t\tIteration " << bagNo + 1 << " out of " << ti.bagN << endl;
		CGrove grove(ti.minAlpha, ti.maxTiGN, ti.interaction);
		grove.trainLayered();
		for(int itemNo = 0; itemNo < validN; itemNo++)
			predsumsV[itemNo] += grove.predict(itemNo, VALID);

		if(!modelFName.empty())
			grove.save(modelFName.c_str()); 
	}

	//calculate predictions of the whole ensemble on the validation set
	doublev predictions(validN); 
	for(int itemNo = 0; itemNo < validN; itemNo++)
		predictions[itemNo] = predsumsV[itemNo] / ti.bagN;

	if(ti.rms)
		return rmse(predictions, validTar);
	else
		return roc(predictions, validTar);	
}

//runs Layered Groves repeatN times, returns average performance and standard deviation
//saves the model from the last run
double meanLG(INDdata& data, TrainInfo ti, int repeatN, double& resStd, string modelFName)
{
	doublev resVals(repeatN);
	int repeatNo;
	cout << endl << "Estimating distribution of model performance" << endl;
	for(repeatNo = 0; repeatNo < repeatN; repeatNo++)
	{
		cout << "\tTraining model " << repeatNo + 1 << " out of " << repeatN << endl;		
		if(repeatNo == repeatN - 1)
			resVals[repeatNo] = layeredGroves(data, ti, modelFName); //save the last model
		else
			resVals[repeatNo] = layeredGroves(data, ti, string(""));
	}

	//calculate mean
	double resMean = 0;
	for(repeatNo = 0; repeatNo < repeatN; repeatNo++)
		resMean += resVals[repeatNo];
	resMean /= repeatN;

	//calculate standard deviation
	resStd = 0;
	for(repeatNo = 0; repeatNo < repeatN; repeatNo++)
		resStd += (resMean - resVals[repeatNo])*(resMean - resVals[repeatNo]);
	resStd /= repeatN;
	resStd = sqrt(resStd);

	return resMean;
}

//implementation for erase for reverse iterator
void rerase(intv& vec, intv::reverse_iterator& reverse)
{
	intv::iterator straight = vec.begin() + ((&*reverse) - (&*vec.begin()));
	reverse++;
	vec.erase(straight);
}

