
LIBPATH := /usr/local/lib
TFLIB := libtensorflow-cpu-linux-x86_64-2.3.0.tar.gz

.PHONY: build

$(LIBPATH)/libtensorflow.so:
	@echo "Downloading: $(TFLIB)"
	wget -c -P /usr/src "https://storage.googleapis.com/tensorflow/libtensorflow/$(TFLIB)"
	tar -C /usr/local -xzf /usr/src/$(TFLIB)

build: $(LIBPATH)/libtensorflow.so
	LIBRARY_PATH=$(LIBPATH) go build -compiler=gccgo -gccgoflags='-fuse-ld=gold' tictactoe.go

clean:
	rm tictactoe