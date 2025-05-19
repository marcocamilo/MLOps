# Project Identification
PROJECT_CODE ?= abc

# Job Configuration
JOB_TYPE ?= batch-runner
DOCKERFILE ?= ./Dockerfile

# Infrastructure Configuration
PROJECT_ID ?= project_id
PROJECTS ?= project_ids
REGION ?= region
ARTIFACT_REPOSITORY ?= repo

# Derived Names
JOB_NAME = $(PROJECT_CODE)-$(JOB_TYPE)
IMAGE_NAME = $(JOB_NAME)

# Full URIs
DOCKER_URI = europe-docker.pkg.dev/$(PROJECT_ID)/$(ARTIFACT_REPOSITORY)/$(PROJECT_CODE)/$(IMAGE_NAME):latest

# Service account - uncomment if needed
# SERVICE_ACCOUNT_NAME = $(PROJECT_CODE)-service-account
# SERVICE_ACCOUNT_EMAIL = $(SERVICE_ACCOUNT_NAME)@$(PROJECT_ID).iam.gserviceaccount.com

# Job configuration parameters
MEMORY ?= 2Gi
CPU ?= 1
TASK_TIMEOUT ?= 3600
TASKS ?= 1
MAX_RETRIES ?= 3
VPC_CONNECTOR ?=
ENVIRONMENT_VARS ?=

# ──────────────────────────────────────────────────────────────────────
#   HELP TARGET                                                  
# ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Cloud Run Job Deployment Makefile"
	@echo "---------------------------"
	@echo "Available targets:"
	@echo "  make docker            - Build and push Docker container to Artifact Registry"
	@echo "  make create-job        - Create/update the Cloud Run job configuration"
	@echo "  make run               - Execute the Cloud Run job"
	@echo "  make delete            - Delete the Cloud Run job"
	@echo "  make logs              - View logs from the most recent job execution"
	@echo ""
	@echo "  make deploy            - Build, push, and create job (docker + create-job)"
	@echo "  make deploy-run        - Build, push, create job and run it (docker + create-job + run)"
	@echo "  make cleanup           - Delete job and related resources"

# ──────────────────────────────────────────────────────────────────────
#   Step 1: Build and push Docker container to Artifact Registry              
# ──────────────────────────────────────────────────────────────────────
.PHONY: docker
docker:
	@echo "📤 Building and pushing Docker container to Artifact Registry..."
	docker build -t $(DOCKER_URI) -f $(DOCKERFILE) .
	docker push $(DOCKER_URI)
	@echo "✅ Docker container pushed successfully to Artifact Registry."

# ──────────────────────────────────────────────────────────────────────
#   Step 2: Create/update the Cloud Run job                            
# ──────────────────────────────────────────────────────────────────────
.PHONY: create-job
create-job:
	@echo "🔧 Creating/updating Cloud Run job configuration..."
	
	# Create or update the job configuration
	gcloud run jobs create $(JOB_NAME) \
	  --project=$(PROJECT_ID) \
	  --region=$(REGION) \
	  --image=$(DOCKER_URI) \
	  --memory=$(MEMORY) \
	  --cpu=$(CPU) \
	  --task-timeout=$(TASK_TIMEOUT)s \
	  --tasks=$(TASKS) \
	  --max-retries=$(MAX_RETRIES) \
	  $(if $(SERVICE_ACCOUNT_EMAIL),--service-account=$(SERVICE_ACCOUNT_EMAIL),) \
	  $(if $(VPC_CONNECTOR),--vpc-connector=$(VPC_CONNECTOR),) \
	  $(if $(ENVIRONMENT_VARS),--set-env-vars=$(ENVIRONMENT_VARS),)
	
	@echo "✅ Cloud Run job configuration created/updated!"

# ──────────────────────────────────────────────────────────────────────
#   Step 3: Execute the Cloud Run job                                  
# ──────────────────────────────────────────────────────────────────────
.PHONY: run
run:
	@echo "🚀 Executing Cloud Run job..."
	
	# Execute the job
	gcloud run jobs execute $(JOB_NAME) \
	  --project=$(PROJECT_ID) \
	  --region=$(REGION)
	
	@echo "✅ Job execution started! Use 'make logs' to follow execution."

# ──────────────────────────────────────────────────────────────────────
#   View logs from most recent job execution                           
# ──────────────────────────────────────────────────────────────────────
.PHONY: logs
logs:
	@echo "📋 Fetching logs from most recent job execution..."
	
	# Get logs from the most recent execution
	gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=$(JOB_NAME) AND resource.labels.location=$(REGION)" \
	  --project=$(PROJECT_ID) \
	  --limit=100 \
	  --format="value(textPayload)"
	
	@echo "✅ End of logs."

# ──────────────────────────────────────────────────────────────────────
#   DELETE JOB                                                        
# ──────────────────────────────────────────────────────────────────────
.PHONY: delete
delete:
	@echo "🗑️ Deleting Cloud Run job..."
	
	# Delete the job
	gcloud run jobs delete $(JOB_NAME) \
	  --project=$(PROJECT_ID) \
	  --region=$(REGION) \
	  --quiet
	
	@echo "✅ Cloud Run job deleted!"

# ──────────────────────────────────────────────────────────────────────
#   CLEANUP (DELETE EVERYTHING)                                        
# ──────────────────────────────────────────────────────────────────────
.PHONY: cleanup
cleanup: delete
	@echo "🧹 Removing container image from Artifact Registry..."
	
	# Delete the container image
	-gcloud artifacts docker images delete $(DOCKER_URI) \
	  --project=$(PROJECT_ID) \
	  --quiet
	
	@echo "✅ Complete cleanup finished - job and container image deleted!"

# ──────────────────────────────────────────────────────────────────────
#   COMBINED TARGETS                                                  
# ──────────────────────────────────────────────────────────────────────
.PHONY: deploy deploy-run
deploy: docker create-job
deploy-run: deploy run

