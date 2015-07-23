# ag_predict_stream
This is a standalone command for running Additive Groves models on hadoop.

Usage: ag_predict -r _attr_file_ [-m _model_file_name_] [-z max_lines_read_in_a_batch] < test_file > predictions_attached_to_test_data


to compile in linux:

ag_predict_stream/AdditiveGroves 
% gmake --makefile Makefile
