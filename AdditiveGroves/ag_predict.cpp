//Additive Groves / ag_predict.cpp: main function of executable ag_predict. Streamed version: data comes from STDIN, predictions go to STDOUT. No log output.
//Output format: predictions in the first column followed by original test data set columns	
//
//(c) Daria Sorokina

#include "Grove.h"
#include "TrainInfo.h"
#include "functions.h"
#include "ErrLogStream.h"

#include <errno.h>

//ag_predict -r _attr_file_ [-m _model_file_name_] [-z max_lines_in_a_batch] < test_data > predictions_attached_to_test_data
int main(int argc, char* argv[])
{	 
	try{

	//1. Set default values of parameters
	string modelFName = "model.bin";	//name of the input file for the model
	int maxLines = 10000;	//number of lines read in a single batch and kept in memory

	TrainInfo ti;

	//2. Set parameters from command line
	//check that the number of arguments is even (flags + value pairs)
	if(argc % 2 == 0)
		throw INPUT_ERR;
	//convert input parameters to string from char*
	stringv args(argc); 
	for(int argNo = 0; argNo < argc; argNo++)
		args[argNo] = string(argv[argNo]);

	//parse and save input parameters
	//indicators of presence of required flags in the input
	bool hasAttr = false;
	
	for(int argNo = 1; argNo < argc; argNo += 2)
	{
		if(!args[argNo].compare("-m"))
			modelFName = args[argNo + 1];
		else if (!args[argNo].compare("-z"))
			maxLines = atoiExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-r"))
		{
			ti.attrFName = args[argNo + 1];
			hasAttr = true;
		}
		else
			throw INPUT_ERR;
	}

//2. Load data
	bool firstBatch = true;
	while(cin.gcount() || firstBatch)
	{
		firstBatch = false;
		INDdata data("", "", "STDIN", ti.attrFName.c_str(), maxLines);
		CGrove::setData(data);
		CTreeNode::setData(data);

	//3. Open model file, read its header
		fstream fmodel(modelFName.c_str(), ios_base::binary | ios_base::in);
		fmodel.read((char*) &ti.mode, sizeof(enum AG_TRAIN_MODE));
		if(ti.mode == FAST)
		{//skip information about fast training - it is not used in this command
			int dirN = 0;
			fmodel.read((char*) &dirN, sizeof(int));
			bool dirStub = false;
			for(int dirNo = 0; dirNo < dirN; dirNo++)
				fmodel.read((char*) &dirStub, sizeof(bool));
		}
		fmodel.read((char*) &ti.maxTiGN, sizeof(int));
		fmodel.read((char*) &ti.minAlpha, sizeof(double));
		if(fmodel.fail() || (ti.maxTiGN < 1))
			throw MODEL_ERR;

	//4. Load models, get predictions	
		doublev testTar;
		int testN = data.getTargets(testTar, TEST);
		doublev preds(testN, 0);

		ti.bagN = 0;
		while(fmodel.peek() != char_traits<char>::eof())
		{//load next Grove in the ensemble 
			ti.bagN++;
			CGrove grove(ti.minAlpha, ti.maxTiGN);
			grove.load(fmodel);

			//get predictions, add them to predictions of previous models
			for(int itemNo = 0; itemNo < testN; itemNo++)
				preds[itemNo] += grove.predict(itemNo, TEST);
		}

		//get bagged predictions of the ensemble
		for(int itemNo = 0; itemNo < testN; itemNo++)
			preds[itemNo] /= ti.bagN;
	
	//5. Output predictions followed by corresponding data line into the output stream
		stringv& textData = *(data.getText());
		for(int itemNo = 0; itemNo < testN; itemNo++)
			cout << preds[itemNo] << "\t" << textData[itemNo] << endl;
	}

	}catch(TE_ERROR err){
		te_errMsg((TE_ERROR)err);
		return 1;
	}catch(AG_ERROR err){
		ErrLogStream errlog;
		switch(err) 
		{
			case INPUT_ERR:
				errlog << "Usage: ag_predict -r _attr_file_name_ [-m _model_file_name_] [-o _output_file_name_] [-c rms|roc]\n";
				break;
			default:
				throw err;
		}
		return 1;
	}catch(exception &e){
		ErrLogStream errlog;
		string errstr(e.what());
		errlog << "Error: " << errstr << "\n";
		return 1;
	}catch(...){
		string errstr = strerror(errno);
		ErrLogStream errlog;
		errlog << "Error: " << errstr << "\n";
		return 1;
	}
}
