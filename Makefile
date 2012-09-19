#***************************************************************************
#                               Makefile
#                          -------------------
#                           W. Michael Brown
#
#  _________________________________________________________________________
#    Mike's Build for Geryon Unified Coprocessor Library (UCL)
#
#  _________________________________________________________________________
#
#    begin                : Thu November 12 2009
#    copyright            : (C) 2009 by W. Michael Brown
#    email                : brownw@ornl.gov
# ***************************************************************************/

# Include specific makefile 
ifndef ARCH
ifeq (,$(findstring clean, $(MAKECMDGOALS)))
ifeq (,$(findstring setup, $(MAKECMDGOALS)))
ifeq (,$(findstring dist, $(MAKECMDGOALS)))
ifeq (,$(findstring dox, $(MAKECMDGOALS)))
$(error Please use: make ARCH=<your_arch>)
else
DOX = doxygen
endif
endif
endif
endif
else
include Makefile.$(ARCH)
endif

# Target binary, object and source(include) directory 
BIN_DIR   = bin
OBJ_DIR   = obj
INC_DIR   = ./include

# Headers for UCL, NVIDIA and OpenCL
UCL_H     = $(notdir $(wildcard $(INC_DIR)/ucl*.h))
NVD_H     = $(notdir $(wildcard $(INC_DIR)/nvd*.h) $(UCL_H))
NVC_H     = $(notdir $(wildcard $(INC_DIR)/nvc*.h) $(UCL_H))
OCL_H     = $(notdir $(wildcard $(INC_DIR)/ocl*.h) $(UCL_H))

# Define your object dependencies here
nvc_example  = nvc_example.o example_kernel.o 
nvd_example  = nvd_example.o  
ocl_example  = ocl_example.o  
nvc_get_devices = nvc_get_devices.o
nvd_get_devices = nvd_get_devices.o
ocl_get_devices = ocl_get_devices.o

# Define your executables here
NVC_EXE = nvc_get_devices nvc_example 
NVD_EXE = nvd_get_devices nvd_example 
OCL_EXE = ocl_get_devices ocl_example 

# Generate binary targets
NVC_BIN   = $(addprefix $(BIN_DIR)/,$(NVC_EXE))
NVD_BIN   = $(addprefix $(BIN_DIR)/,$(NVD_EXE))
OCL_BIN   = $(addprefix $(BIN_DIR)/,$(OCL_EXE))

# Compile lines
NVC      += $(NVC_FLAGS) $(NVC_INCS) -I$(INC_DIR)
OCL      += $(OCL_FLAGS) $(OCL_INCS) -I$(INC_DIR)
CPP      += $(CPP_FLAGS) $(CPP_INCS) -I$(INC_DIR)

# Prevent from deleating itermediate files
.SECONDARY: $(OBJ)
.PHONY: test all setup clean
.NOTPARALLEL: setup clean

# Default makefile target
all: setup $(NVC_BIN) $(NVD_BIN) $(OCL_BIN)

define nvc_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(NVC) -o $$@ $$^ -DUCL_CUDART $$(NVC_LIBS)
endef

define nvd_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(NVC) -o $$@ $$^ -DUCL_CUDADR $$(NVC_LIBS)
endef

define ocl_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(OCL) -o $$@ $$^ -DUCL_OPENCL $$(OCL_LIBS)
endef

$(foreach p,$(NVC_EXE),$(eval $(call nvc_template,$(p))))
$(foreach p,$(NVD_EXE),$(eval $(call nvd_template,$(p))))
$(foreach p,$(OCL_EXE),$(eval $(call ocl_template,$(p))))

# Add dependencies
OBJ_DEP = $(OBJ:%.o=%.d)
-include $(OBJ_DEP)

# Rules for compiling the example object files
$(OBJ_DIR)/nvc_%.o: examples/%.cpp 
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDART

$(OBJ_DIR)/nvd_%.o: examples/%.cpp
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDADR

$(OBJ_DIR)/ocl_%.o: examples/%.cpp
	$(OCL) -MMD -o $@ -c $< -DUCL_OPENCL

$(OBJ_DIR)/%.o: examples/%.cu
	$(NVC) -DNV_KERNEL -DORDINAL=int -DScalar=float -o $@ -c $<

# Rule for doxygen
dox:
	@echo "Running doxygen"
	$(DOX) Doxyfile

# Rules for setup, dist and clean
setup:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

clean:
	@echo "Clean"
	@rm -rf $(BIN_DIR) $(OBJ_DIR) ./doc/api geryon.*.tar.gz

VERSION = `date +'%y.%j'`

dist: clean
	@echo "Creating version $(VERSION)"
	@echo "Geryon Version $(VERSION)" > VERSION.txt
	@echo "#define GERYON_VERSION \0042$(VERSION)\0042" > $(INC_DIR)/ucl_version.h
	@tar -cz doc examples test include Makefile* > geryon.$(VERSION).tar.gz

# Rules to build the test cases
TEST_DEP  = $(BIN_DIR)/ucl_test 
TEST_DEP += $(BIN_DIR)/ucl_test_kernel.ptx $(BIN_DIR)/ucl_test_kernel_d.ptx
TEST_DEP += $(BIN_DIR)/ucl_test_kernel.cubin $(OBJ_DIR)/ucl_test_kernel.o

$(OBJ_DIR)/%.o: test/%.cpp 
	$(CPP) -MMD $(NVC_INCS) $(OCL_INCS) -o $@ -c $<

$(BIN_DIR)/%.ptx: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -ptx -o $@ $<	

$(BIN_DIR)/%_d.ptx: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=double -ptx -o $@ $<	

$(BIN_DIR)/%.cubin: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -cubin -o $@ $<	

$(OBJ_DIR)/%.o: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -cubin -o $@ $<

$(BIN_DIR)/ucl_test: $(OBJ_DIR)/ucl_test.o
	$(CPP) -o $@ $^ $(OCL_LIBS) $(CPP_LIBS) $(NVC_LIBS)

test: setup $(TEST_DEP) 
	cp test/*.cu $(BIN_DIR)/
	cp test/*.h  $(BIN_DIR)/
	$(BIN_DIR)/ucl_test $(BIN_DIR)


