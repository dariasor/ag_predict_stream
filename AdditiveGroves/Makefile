

SHAREDDIR=../shared
BINDIR=../Bin
LIBDIR=../ThreadPool
CXXFLAGS = -I$(SHAREDDIR) -I$(LIBDIR) #-static
OBJS = Grove.o $(SHAREDDIR)/SplitInfo.o  $(SHAREDDIR)/INDdata.o $(SHAREDDIR)/TreeNode.o ag_functions.o $(SHAREDDIR)/functions.o $(SHAREDDIR)/LogStream.o $(LIBDIR)/thread_pool.o
PGMS = ag_predict_stream
PGMOBJS = ag_predict.o
LIBS = -lpthread

all: $(PGMS)

$(PGMS): | $(BINDIR)

clean:
	rm $(OBJS)
	rm $(PGMOBJS)
	rm $(PGMS)

.cpp.o:
	g++ -O3 $(CXXFLAGS) -c $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

ag_predict_stream: ag_predict.o $(OBJS)
	g++ -O3 -o $(BINDIR)/ag_predict_stream ag_predict.o $(OBJS) $(LIBS)


