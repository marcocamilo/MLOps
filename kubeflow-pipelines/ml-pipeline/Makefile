COMPONENT_DIR ?= src
COMPONENT_FILES ?= generate.py validate.py preprocess.py train.py evaluate.py
PUSH_IMAGE ?= true

# Internal flags
KFP_BUILD_CMD = kfp component build $(COMPONENT_DIR) --component-filepattern
PUSH_FLAG = $(if $(filter true,$(PUSH_IMAGE)),--push-image,--no-push-image)

# Default target
all: components pipeline clean

# Help command
help:
	@echo "Kubeflow Pipeline Makefile"
	@echo "---------------------------"
	@echo "Available targets:"
	@echo "  make components  - Build Kubeflow pipeline components from Python files in \`$(COMPONENT_DIR)\`"
	@echo "  make pipeline      - Run the main pipeline script (pipeline.py)"
	@echo "  make clean             - Clean up generated build artifacts and temporary files"
	@echo "  make all               - Run components, pipeline, and clean in order"
	@echo ""
	@echo "Variables:"
	@echo "  COMPONENT_DIR=<dir>        - Directory containing component scripts (default: src)"
	@echo "  COMPONENT_FILES=<files>    - Python files to convert into components (default: data.py train.py)"
	@echo "  PUSH_IMAGE=true|false      - Whether to push component images to registry (default: true)"

# Build all component YAMLs
components:
	@echo "🏗️ Building Kubeflow components..."
	@for file in $(COMPONENT_FILES); do \
		echo "👉 Building component: $$file"; \
		$(KFP_BUILD_CMD) $$file $(PUSH_FLAG); \
	done
	@echo "👍 Components built"

# Run the pipeline
pipeline:
	@echo "⚙️ Running pipeline..."
	python pipeline.py
	@echo "✅ Pipeline done!"

# Clean target (optional)
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(COMPONENT_DIR)/__pycache__
	@rm -rf $(COMPONENT_DIR)/component_metadata
	@rm -f  $(COMPONENT_DIR)/Dockerfile
	@rm -f  $(COMPONENT_DIR)/kfp_config.ini
	@rm -f  $(COMPONENT_DIR)/runtime-requirements.txt

	# @find $(COMPONENT_DIR) -maxdepth 1 -name '*.py' \
	# 	! -name 'mlflow_utils.py' \
	# 	$(foreach file,$(COMPONENT_FILES),! -name '$(file)' ) \
	# 	-exec rm -f {} \;

	@echo "✨ Component directory cleaned"
