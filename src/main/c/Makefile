CC=zig cc
UNAME_S := $(shell uname -s)

JAVA_INCLUDE := $(shell java -XshowSettings:properties -version 2>&1 | grep 'java.home' | awk '{print $$3 "/include"}')

ifeq ($(UNAME_S),Linux)
    PLATFORM=linux
    SOEXT=so
    SOFLAGS=-shared -fPIC
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM=darwin
    SOEXT=dylib
    SOFLAGS=-dynamiclib
endif

JNI_CFLAGS=-I$(JAVA_INCLUDE) -I$(JAVA_INCLUDE)/$(PLATFORM)
JNI_LIB=libjtinn.$(SOEXT)

all: $(JNI_LIB)

$(JNI_LIB): jtinn.c
	$(CC) $(JNI_CFLAGS) $(SOFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(JNI_LIB)
