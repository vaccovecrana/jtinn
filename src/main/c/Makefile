CC=zig cc
UNAME_S := $(shell uname -s)

JAVA_INCLUDE := $(shell java -XshowSettings:properties -version 2>&1 | grep 'java.home' | awk '{print $$3 "/include"}')

ifeq ($(UNAME_S),Linux)
    PLATFORM=linux
    SOEXT=so
    SOFLAGS=-shared -fPIC
    TARGET_FLAG=-target x86_64-linux-gnu
    LDFLAGS += -s  # Strip symbols (equivalent to --strip-all)
    CFLAGS += -Os  # Optimize for size
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM=darwin
    SOEXT=dylib
    SOFLAGS=-dynamiclib
    LDFLAGS += -s
    CFLAGS += -Os
endif

JNI_CFLAGS=-I$(JAVA_INCLUDE) -I$(JAVA_INCLUDE)/$(PLATFORM)
JNI_LIB=libjtinn.$(SOEXT)

all: $(JNI_LIB)

$(JNI_LIB): jtinn.c
	$(CC) $(TARGET_FLAG) $(JNI_CFLAGS) $(SOFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(JNI_LIB)